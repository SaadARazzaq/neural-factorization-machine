import tensorflow as tf
import numpy as np
import math

class NFMClassifier():
    """
    Neural Factorization Machine Classifier
    Original Paper: He, X., & Chua, T.-S. (2017). Neural Factorization Machines for Sparse Predictive Analytics
    """
    
    def __init__(self, inp_dim, emb_dim=16, hidden_units=[64, 32], n_classes=2, 
                 keep_prob=0.8, use_gpu=False):
        self._n_classes = n_classes
        self._keep_prob = keep_prob
        self._use_gpu = use_gpu
        self._inp_dim = inp_dim
        self._emb_dim = emb_dim
        self._hidden_units = hidden_units
        
        if use_gpu:
            with tf.device('/device:GPU:0'):
                self._build_graph()
        else:
            with tf.device('/device:CPU:0'):
                self._build_graph()

    def _build_graph(self):
        """Build the computational graph for Neural Factorization Machine"""
        self._keep_prob_tensor = tf.placeholder(tf.float32)
        self._is_training = tf.placeholder(tf.bool)
        self._X = tf.placeholder(shape=[None, self._inp_dim], dtype=tf.float32)
        
        # Input normalization
        self._X_norm = tf.contrib.layers.batch_norm(
            self._X, 
            is_training=self._is_training
        )

        # NFM Components
        self._embedding = self._embedding_layer(
            self._X_norm, 
            inp_dim=self._inp_dim, 
            emb_dim=self._emb_dim, 
            name="embedding"
        )
        self._bipooled = self._bi_interaction_pooling(self._embedding)
        self._bipooled_dropout = tf.nn.dropout(
            self._bipooled, 
            keep_prob=self._keep_prob_tensor
        )

        # Deep Neural Network Component
        self._deep_output = self._build_deep_network(self._bipooled_dropout)

        # Linear terms and global bias
        self._global_bias = tf.get_variable(
            name="global_bias", 
            shape=[self._n_classes], 
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer()
        )

        self._W = tf.get_variable(
            name="W", 
            shape=[self._inp_dim, self._n_classes], 
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer()
        )

        # NFM equation: ŷ(x) = w₀ + Σwᵢxᵢ + f_MLP(ΣΣ⟨vᵢ,vⱼ⟩xᵢxⱼ)
        self._op = (
            self._deep_output +  # Deep learned interactions
            tf.matmul(self._X_norm, self._W) +  # Linear terms  
            self._global_bias  # Global bias
        )

        self._op_prob = tf.nn.softmax(self._op, name="probabilities")

    def _build_deep_network(self, bipooled):
        """Build multi-layer perceptron on top of bi-interaction pooling"""
        current_layer = bipooled
        
        for i, units in enumerate(self._hidden_units):
            current_layer = self._fully_connected_layer(
                current_layer,
                inp_dim=current_layer.get_shape()[-1].value,
                op_dim=units,
                name=f"fc_{i+1}",
                activation=tf.nn.relu
            )
            current_layer = tf.nn.dropout(current_layer, keep_prob=self._keep_prob_tensor)
        
        # Output layer
        output_layer = self._fully_connected_layer(
            current_layer,
            inp_dim=current_layer.get_shape()[-1].value,
            op_dim=self._n_classes,
            name="output",
            activation=None
        )
        
        return output_layer

    def _fully_connected_layer(self, x, inp_dim, op_dim, name, activation=None):
        """Create a fully connected layer with batch normalization"""
        W = tf.get_variable(
            f"W_{name}", 
            shape=[inp_dim, op_dim],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        b = tf.get_variable(
            f"b_{name}", 
            shape=[op_dim],
            initializer=tf.zeros_initializer()
        )
        
        z = tf.matmul(x, W) + b
        
        if activation is not None:
            a = activation(z)
            # Batch normalization for hidden layers
            if activation != tf.nn.softmax:  # Don't apply to output layer
                a = tf.layers.batch_normalization(a, training=self._is_training)
            return a
        return z

    def _embedding_layer(self, x, inp_dim, emb_dim, name):
        """Create embedding matrix V ∈ R^(n×k) for latent vectors"""
        V = tf.get_variable(
            name="embed_" + name, 
            shape=[inp_dim, emb_dim],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        x_reshape = tf.expand_dims(x, 2)
        x_tiled = tf.tile(x_reshape, multiples=(1, 1, emb_dim))
        return tf.multiply(x_tiled, V)

    def _bi_interaction_pooling(self, embedding):
        """
        Bi-Interaction Pooling Layer
        Efficiently computes: ½[(Σvᵢxᵢ)² - Σ(vᵢxᵢ)²]
        """
        sum_square = tf.reduce_sum(embedding, axis=1) * tf.reduce_sum(embedding, axis=1)
        square_sum = tf.reduce_sum(embedding * embedding, axis=1)
        return 0.5 * (sum_square - square_sum)

    def fit(self, X, y, num_epoch=100, batch_size=32, validation_data=None, 
            weight_save_path=None, weight_load_path=None, verbose=1):
        """
        Train the Neural Factorization Machine model
        """
        self._y = tf.placeholder(tf.float32, shape=[None, self._n_classes])
        
        # Loss function
        self._mean_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self._op, 
                labels=self._y
            )
        )
        
        # Optimizer with update ops for batch norm
        self._optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self._train_step = self._optimizer.minimize(self._mean_loss)
        
        # Initialize session
        self._sess = tf.Session()
        
        # Load weights if provided
        if weight_load_path is not None:
            loader = tf.train.Saver()
            loader.restore(sess=self._sess, save_path=weight_load_path)
            if verbose > 0:
                print("Weights loaded successfully from:", weight_load_path)
        else:
            self._sess.run(tf.global_variables_initializer())
        
        # Training loop
        if num_epoch > 0:
            if verbose > 0:
                print(f'Training Neural Factorization Machine for {num_epoch} epochs')
            
            self._train_loop(
                X, y, validation_data, num_epoch, batch_size, 
                weight_save_path, verbose
            )

    def _train_loop(self, X, y, validation_data, num_epoch, batch_size, 
                   weight_save_path, verbose):
        """Main training loop"""
        saver = tf.train.Saver()
        best_val_loss = float('inf')
        
        for epoch in range(num_epoch):
            # Shuffle data
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_losses = []
            for i in range(0, len(X), batch_size):
                batch_X = X_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]
                
                feed_dict = {
                    self._X: batch_X,
                    self._y: batch_y,
                    self._is_training: True,
                    self._keep_prob_tensor: self._keep_prob
                }
                
                _, loss = self._sess.run(
                    [self._train_step, self._mean_loss], 
                    feed_dict=feed_dict
                )
                epoch_losses.append(loss)
            
            # Validation
            if validation_data is not None:
                val_loss, val_acc = self.evaluate(
                    validation_data[0], 
                    validation_data[1], 
                    batch_size
                )
                
                if verbose > 0:
                    print(f"Epoch {epoch+1}/{num_epoch}, "
                          f"Train Loss: {np.mean(epoch_losses):.4f}, "
                          f"Val Loss: {val_loss:.4f}, "
                          f"Val Acc: {val_acc:.4f}")
                
                # Save best model
                if val_loss < best_val_loss and weight_save_path is not None:
                    best_val_loss = val_loss
                    saver.save(self._sess, weight_save_path)
                    if verbose > 1:
                        print(f"Model saved to {weight_save_path}")
            else:
                if verbose > 0:
                    print(f"Epoch {epoch+1}/{num_epoch}, "
                          f"Train Loss: {np.mean(epoch_losses):.4f}")

    def predict(self, X):
        """Make predictions on new data"""
        feed_dict = {
            self._X: X,
            self._is_training: False,
            self._keep_prob_tensor: 1.0
        }
        probabilities = self._sess.run(self._op_prob, feed_dict=feed_dict)
        return np.argmax(probabilities, axis=1)

    def predict_proba(self, X):
        """Predict class probabilities"""
        feed_dict = {
            self._X: X,
            self._is_training: False,
            self._keep_prob_tensor: 1.0
        }
        return self._sess.run(self._op_prob, feed_dict=feed_dict)

    def evaluate(self, X, y, batch_size=32):
        """Evaluate model on given data"""
        total_correct = 0
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i + batch_size]
            batch_y = y[i:i + batch_size]
            
            feed_dict = {
                self._X: batch_X,
                self._y: batch_y,
                self._is_training: False,
                self._keep_prob_tensor: 1.0
            }
            
            loss, probs = self._sess.run(
                [self._mean_loss, self._op_prob], 
                feed_dict=feed_dict
            )
            
            predictions = np.argmax(probs, axis=1)
            true_labels = np.argmax(batch_y, axis=1)
            correct = np.sum(predictions == true_labels)
            
            total_correct += correct
            total_loss += loss * len(batch_X)
            num_batches += 1
        
        accuracy = total_correct / len(X)
        avg_loss = total_loss / len(X)
        
        return avg_loss, accuracy

    def __del__(self):
        """Clean up TensorFlow session"""
        if hasattr(self, '_sess'):
            self._sess.close()
