import numpy as np
from threading import Thread, Event
from tensorflow.keras.utils import plot_model

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLabel, QLineEdit, QSpinBox, QPushButton, QHBoxLayout, QSpacerItem, QSizePolicy, QDialog, QMessageBox
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont, QPixmap

from model import create_model
from progress_window import ProgressWindow
from utils import generate_training_data, context_size, global_training_history, model


# Create the model
model = create_model(1, context_size)

# PyQt5 GUI Class
class App(QWidget):
    update_progress_signal = pyqtSignal(int)
    update_info_signal = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.title = 'LSTM Visualization Tool'
        self.initUI()
        self.X, self.y = None, None
        self.training_thread = None
        self.training_event = Event()
        self.context_size = context_size
        self.progress_window = ProgressWindow() 
        self.update_progress_signal.connect(self.update_progress)
        self.update_info_signal.connect(self.update_training_info)
        self.is_model_trained = False
        
    def initUI(self):
        self.setWindowTitle(self.title)
        
        main_layout = QVBoxLayout()     

        main_layout.addWidget(self.create_input_group())

        # Spacer before Status Label
        spacer_before_status = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        main_layout.addSpacerItem(spacer_before_status)
        
        self.status_label = QLabel("<b>Status:</b> Ready (model untrained)")
        self.status_label.setFont(QFont('Arial', 10))
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        main_layout.addWidget(self.create_output_group())        
        main_layout.addWidget(self.create_control_buttons())

        info_layout = QHBoxLayout()
        info_layout.addWidget(self.create_network_info_group())
        info_layout.addWidget(self.create_training_info_group())
        main_layout.addLayout(info_layout)
        
        self.setLayout(main_layout)
        
    def create_input_group(self):
        input_group = QGroupBox("Input Data")
        
        input_layout = QFormLayout()

        self.input_field = QLineEdit(self)
        self.input_field.textChanged.connect(self.update_original_sequence)
        input_layout.addRow("Enter Sequence (comma-separated, e.g., 0,1,0,1):", self.input_field)

        self.epoch_input = QSpinBox(self)
        self.epoch_input.setMinimum(1)
        self.epoch_input.setMaximum(1000)
        self.epoch_input.setValue(10)
        input_layout.addRow("Set Number of Training Epochs:", self.epoch_input)

        input_group.setLayout(input_layout)
        return input_group

    def create_control_buttons(self):
        control_group = QGroupBox("")
        main_layout = QVBoxLayout()  # Main vertical layout

        # Set control panel title and center it
        title_label = QLabel("Model Training Control Panel")
        title_label.setFont(QFont('Arial', 10))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Horizontal layout for control buttons
        control_button_layout = QHBoxLayout()

        # Start Training Button
        self.start_button = QPushButton('Start')
        self.start_button.setStyleSheet("background-color: #CADEDB")
        self.start_button.clicked.connect(self.start_training)
        control_button_layout.addWidget(self.start_button)

        # Pause Training Button
        self.stop_button = QPushButton('Pause')
        self.stop_button.setStyleSheet("background-color: #FCF4D3")
        self.stop_button.clicked.connect(self.stop_training)
        control_button_layout.addWidget(self.stop_button)
        
        # Continue Training Button
        self.continue_button = QPushButton('Continue')
        self.continue_button.setStyleSheet("background-color: #FCF4D3")
        self.continue_button.clicked.connect(self.continue_training)
        control_button_layout.addWidget(self.continue_button)
        
        # Reset Training Button
        self.reset_button = QPushButton('Reset')
        self.reset_button.setStyleSheet("background-color: #F1D7D3")
        self.reset_button.clicked.connect(self.reset_training)
        control_button_layout.addWidget(self.reset_button)

        # Add control buttons layout to the main layout
        main_layout.addLayout(control_button_layout)

        # Horizontal layout for Show Progress Button
        show_progress_layout = QHBoxLayout()
        show_progress_layout.addStretch()

        # Show Progress Button
        self.progress_button = QPushButton('Show Training Progress')
        self.progress_button.setStyleSheet("background-color: #CADEDB")
        self.progress_button.clicked.connect(self.show_progress)
        show_progress_layout.addWidget(self.progress_button)

        show_progress_layout.addStretch()
        main_layout.addLayout(show_progress_layout)

        control_group.setLayout(main_layout)
        return control_group

    def create_output_group(self):
        output_group = QGroupBox("Output Data")
        
        # Main horizontal layout
        main_layout = QHBoxLayout()

        # Vertical layout for labels
        output_layout = QVBoxLayout()
        self.original_sequence_label = QLabel("Original Sequence: N/A")
        output_layout.addWidget(self.original_sequence_label)
        self.updated_sequence_label = QLabel("Updated Sequence: N/A")
        output_layout.addWidget(self.updated_sequence_label)
        self.prediction_label = QLabel("Prediction will appear here after processing.")
        output_layout.addWidget(self.prediction_label)

        # Add vertical layout to the main horizontal layout
        main_layout.addLayout(output_layout)

        # Spacer to push the button to the right
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        main_layout.addSpacerItem(spacer)

        # Predict button
        self.predict_button = QPushButton('Predict', self)
        # Make 'predict' bold
        self.predict_button.setFont(QFont('Arial', 8, weight=QFont.Bold))

        self.predict_button.setStyleSheet("background-color: #AEE4F6")
        self.predict_button.clicked.connect(self.predict_sequence)
        
        # Add the button to the main horizontal layout
        main_layout.addWidget(self.predict_button)

        output_group.setLayout(main_layout)
        return output_group

    def create_network_info_group(self):
        network_info_group = QGroupBox("Network Information")
        network_info_layout = QVBoxLayout()

        self.network_params_label = QLabel(f"Total Parameters: {model.count_params()}")
        network_info_layout.addWidget(self.network_params_label)
        
        for layer in model.layers:
            layer_info = QLabel(f"Layer: {type(layer).__name__}, "
                                f"Units: {layer.units if hasattr(layer, 'units') else 'N/A'}, "
                                f"Activation: {layer.activation.__name__ if hasattr(layer, 'activation') else 'N/A'}")
            network_info_layout.addWidget(layer_info)

        # Button to visualize the model
        self.visualize_model_button = QPushButton('Show Model Architecture', self)
        self.visualize_model_button.setStyleSheet("background-color: #CADEDB")
        self.visualize_model_button.clicked.connect(self.visualize_model)
        self.visualize_model_button.setToolTip("Click to view the structure of the LSTM model")
        network_info_layout.addWidget(self.visualize_model_button)

        network_info_group.setLayout(network_info_layout)
        return network_info_group

    def create_training_info_group(self):
        training_info_group = QGroupBox("Training Information")
        training_info_layout = QVBoxLayout()

        self.train_status_label = QLabel("Status: Model untrained")
        training_info_layout.addWidget(self.train_status_label)

        self.epoch_label = QLabel("Epoch: N/A")
        training_info_layout.addWidget(self.epoch_label)
        
        self.loss_label = QLabel("Loss: N/A")
        training_info_layout.addWidget(self.loss_label)
        
        self.accuracy_label = QLabel("Accuracy: N/A")
        training_info_layout.addWidget(self.accuracy_label)

        training_info_group.setLayout(training_info_layout)
        return training_info_group

    def update_training_info(self, info):
        self.loss_label.setText(f"Current Loss: {info.get('loss', 'N/A')}")
        self.accuracy_label.setText(f"Current Accuracy: {info.get('accuracy', 'N/A')}")
        self.epoch_label.setText(f"Current Epoch: {info.get('epoch', 'N/A')}")
        
    def update_progress(self, value):
        if hasattr(self, 'progress_window'):
            self.progress_window.update_progress(value)
            
    def reset_training(self):
        global model, global_training_history
        model = create_model(1, self.context_size)
        self.training_event.set()
        self.clear_plots()

        # Clear training information
        global_training_history = {'loss': [], 'accuracy': []}
        self.train_status_label.setText("Status: Model untrained")
        self.epoch_label.setText("Epoch: N/A")
        self.loss_label.setText("Loss: N/A")
        self.accuracy_label.setText("Accuracy: N/A")
        self.status_label.setText("<b>Status:</b> Ready (training reset)")


    def clear_plots(self):
        if hasattr(self, 'progress_window'):
            self.progress_window.clear_plots()
    
    def visualize_model(self):
        # Save the model plot to a temporary file
        plot_file = './model_plot.png'
        plot_model(model, to_file=plot_file, show_shapes=True, show_layer_names=True)

        # Create a widget to show the image
        image_dialog = QDialog(self)
        image_dialog.setWindowTitle("Model Visualization")
        
        layout = QVBoxLayout()
        image_label = QLabel()
        pixmap = QPixmap(plot_file)
        image_label.setPixmap(pixmap)
        
        # Set "What's This?" text for the image label
        image_label.setWhatsThis("This image shows the architecture of the LSTM model used in the application.")
        layout.addWidget(image_label)
        
        image_dialog.setLayout(layout)
        image_dialog.exec_()
        
    def update_original_sequence(self, text):
        if text:
            self.original_sequence_label.setText(f"Original Sequence: {text}")
        else:
            self.original_sequence_label.setText("Original Sequence: N/A")
    
    def predict_sequence(self):
        input_sequence = self.input_field.text()
        if not input_sequence:
            QMessageBox.warning(self, 'Invalid Input', 'Please enter a sequence.')
            return
        if not self.is_model_trained:
            QMessageBox.warning(self, 'Model Untrained', 'Please train the model before making predictions.')
            return
        try:
            prediction, confidence = self.calculate_prediction(input_sequence)
            if prediction is not None:
                self.train_status_label.setText("Status: Predicting...")
                self.status_label.setText("<b>Status:</b> Predicting...")
                self.original_sequence_label.setText(f"Original Sequence: {input_sequence}")
                updated_sequence = input_sequence + ',' + str(prediction)
                self.updated_sequence_label.setText(f"Updated Sequence: {updated_sequence}")
                # Format confidence as a percentage
                confidence_percent = f"{confidence * 100:.2f}%"
                self.prediction_label.setText(f"Predicted value for the next step: {prediction} (Confidence: {confidence_percent})")
                self.status_label.setText("<b>Status:</b> Prediction done")
        except ValueError as e:
            QMessageBox.warning(self, 'Invalid Input', str(e))
            self.original_sequence_label.setText("Original Sequence: N/A")
            self.updated_sequence_label.setText("Updated Sequence: N/A")

    def calculate_prediction(self, input_sequence):
        try:
            data = [int(x) for x in input_sequence.split(',')]
            if len(data) < self.context_size:
                raise ValueError(f"Input sequence [{len(data)}] is too short for the context size [{context_size}].")
            last_n_elements = data[-self.context_size:]
            processed_input = np.array(last_n_elements).reshape(1, self.context_size, 1)
            prediction_output = model.predict(processed_input).flatten()
            prediction = int(np.round(prediction_output[-1]))

            # Calculate unified confidence score
            if prediction == 1:
                confidence = prediction_output[-1]  # Confidence of prediction being 1
            else:
                confidence = 1 - prediction_output[-1]  # Confidence of prediction being 0

            confidence_percentage = round(confidence, 2)  # Convert to percentage
            print(f"Prediction: {prediction}, Confidence: {confidence_percentage}%")
            return prediction, confidence_percentage
        except ValueError as e:
            raise e

        
    def start_training(self):
        if self.training_thread is None or not self.training_thread.is_alive():
            input_sequence = self.input_field.text()
            if not input_sequence:
                QMessageBox.warning(self, 'Invalid Input', 'Please enter a sequence.')
                return
            self.X, self.y = generate_training_data(input_sequence, self.context_size)

            self.clear_plots()
            self.training_event.clear()
            self.training_thread = Thread(target=lambda: self.train_and_visualize(epochs=self.epoch_input.value(), batch_size=1))
            self.training_thread.start()
            self.train_status_label.setText("Status: training...")
            self.status_label.setText("<b>Status:</b> Training...")

    def stop_training(self):
        self.training_event.set()
        self.status_label.setText("<b>Status:</b> Training paused")
        self.train_status_label.setText("Status: training paused")

    def continue_training(self):
        if self.training_thread is not None and not self.training_thread.is_alive():
            self.training_event.clear()
            self.training_thread = Thread(target=lambda: self.train_and_visualize(epochs=self.epoch_input.value(), batch_size=1, continue_training=True))
            self.training_thread.start()
            self.status_label.setText("<b>Status:</b> Training...")
            self.train_status_label.setText("Status: training continued")

    def reset_training(self):
        global model
        self.clear_plots()
        model = create_model(1, self.context_size)
        # Clear the training dict info
        global global_training_history
        global_training_history = {'loss': [], 'accuracy': []}
        self.training_event.set()
        self.status_label.setText("<b>Status:</b> Ready (training reset)")
        self.train_status_label.setText("Status: training reset")

    def show_progress(self):
        global global_training_history
        self.progress_window.update_plots(None, global_training_history, self.y)
        self.progress_window.show(self.y)

    def train_and_visualize(self, epochs=10, batch_size=1, continue_training=False):
        global global_training_history
        history = global_training_history if continue_training else {'loss': [], 'accuracy': []}
        training_paused = False
        for epoch in range(epochs):
            if self.training_event.is_set():
                training_paused = True
                # Update status label to indicate training is paused
                self.status_label.setText("<b>Status:</b> Training paused")
                continue  # Skip the rest of the loop

            h = model.fit(self.X, self.y, epochs=1, batch_size=batch_size, verbose=2)
            predictions = model.predict(self.X, batch_size=batch_size).flatten()

            history['loss'].append(h.history['loss'][0])
            history['accuracy'].append(h.history['accuracy'][0])

            # Update plots, progress, and training info
            self.progress_window.update_plots(predictions, history, self.y)
            progress = int((epoch + 1) / epochs * 100)
            self.update_progress_signal.emit(progress)
            self.update_training_info({
                'loss': round(h.history['loss'][0], 4),
                'accuracy': round(h.history['accuracy'][0], 4),
                'epoch': epoch + 1
            })

        # Save the training history globally and update the status
        global_training_history = history
        if not training_paused:
            self.status_label.setText("<b>Status:</b> Training finished, ready for prediction")
            self.train_status_label.setText("Status: training finished")
            self.update_progress_signal.emit(100)
            self.is_model_trained = True