# Standard library imports
from threading import Event, Thread

# Third-party imports
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QDoubleValidator
from PyQt5.QtWidgets import (QAction, QComboBox, QDialog, QFileDialog, QFormLayout, QFrame, 
                             QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, 
                             QMenuBar, QProgressBar, QPushButton, QSizePolicy, QSpacerItem, QSpinBox, 
                             QVBoxLayout, QWidget)
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Local application imports
from model import create_model
from progress_window import ProgressWindow
from utils import context_size, generate_training_data, global_training_history, model

# TODO: avoid global variables?

# PyQt5 GUI Class
class App(QMainWindow):
    """
    Main application window for the LSTM Visualization Tool. 
    This class sets up the GUI and manages user interactions.
    """

    # Class-level attributes for PyQt signals
    update_progress_signal = pyqtSignal(int)
    update_info_signal = pyqtSignal(dict)

    def __init__(self):
        """
        Initialize the application window, set up the UI, and prepare the model.
        """
        super(App, self).__init__()

        # Set the window title
        self.title = 'LSTM Visualization Tool'
        self.setWindowTitle(self.title)

        # Initialize attributes for model training and UI components
        self.X, self.y = None, None   # Training data
        self.training_thread = None   # Thread for model training
        self.current_epoch = 0        # Current training epoch
        self.training_event = Event() # Event to manage training control
        self.context_size = context_size  # Context size for the LSTM model
        self.is_model_trained = False     # Flag to track if the model is trained

        # Initialize UI components
        self.initUI()
        self.layer_info_labels = []  # Labels for layer information
        self.progress_window = ProgressWindow()  # Window to show training progress

        # Connect signals to slots
        self.update_progress_signal.connect(self.update_progress)
        self.update_info_signal.connect(self.update_training_info)
        self.update_progress_signal.connect(self.update_training_progress)

        # Create initial model
        self.create_initial_model()
     
    def initUI(self):
        """ Initialize the GUI """
        # Set the central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        self.setWindowTitle(self.title)
        main_layout = QVBoxLayout(central_widget)
        main_layout.addWidget(self.create_input_group())

        # Status Label
        self.status_label = QLabel("<b>Status:</b> Ready (model untrained)")
        self.status_label.setStyleSheet("padding-top: 5px; padding-bottom: 3px;")
        self.status_label.setFont(QFont('Arial', 12))
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
                
        # Add Output Panel
        main_layout.addWidget(self.create_output_group()) 
        # Add Control and Training Info Panels
        main_layout.addWidget(self.create_control_buttons())
        # Combine Network Info and Hyperparameters into one group
        combined_info_layout = QHBoxLayout()
        combined_info_layout.addWidget(self.create_network_info_group())
        combined_info_layout.addWidget(self.create_hyperparameter_input_group())
        combined_group = QGroupBox("")
        combined_group.setLayout(combined_info_layout)
        main_layout.addWidget(combined_group)

        self.setLayout(main_layout)
        self.layout().activate()
        min_size = self.layout().minimumSize()
        self.resize(min_size.width(), min_size.height())
        
        self.init_menu_bar()
           
    def init_menu_bar(self):
        # Create a 'File' or 'Model' menu
        file_menu = self.menuBar().addMenu("&Model")

        # Add 'Save Model' action
        save_action = QAction("&Save Model", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_model)
        file_menu.addAction(save_action)

        # Add 'Load Model' action
        load_action = QAction("&Load Model", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_model)
        file_menu.addAction(load_action)
        
        # Exit action
        exit_action = QAction("&Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = self.menuBar().addMenu("&Help")

        # About action
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.about_dialog)
        help_menu.addAction(about_action)

    def about_dialog(self):
        QMessageBox.about(self, "About LSTM Visualization Tool",
                        "LSTM Visualization Tool\nVersion 1.0\nDeveloped by [Šimon Šmída]")
        
    def create_initial_model(self):
        """ Create the initial model with default parameters """
        global model
        # Default hyperparameters
        num_layers = 1
        units_per_layer = 3
        learning_rate = 0.1
        optimizer = 'adam'
        loss = 'binary_crossentropy'
        self.context_size = context_size  # Update this if you have a default value

        model = create_model(num_layers=num_layers, 
                             units_per_layer=units_per_layer, 
                             learning_rate=learning_rate, 
                             optimizer=optimizer, 
                             loss=loss)
        model.build(input_shape=(1, self.context_size, 1))
    
    def update_model(self):
        """ Update the model based on the hyperparameter inputs """
        try:
            learning_rate = float(self.learning_rate_input.text())
        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', 'Please enter a valid learning rate.')
            return
        num_layers = self.num_layers_input.value()
        units_per_layer = self.units_per_layer_input.value()
        learning_rate = float(self.learning_rate_input.text())
        optimizer = self.optimizer_input.currentText()
        global model
        model = create_model(num_layers=num_layers, units_per_layer=units_per_layer,
                             learning_rate=learning_rate, optimizer=optimizer,
                             loss='binary_crossentropy')
        model.build(input_shape=(1, self.context_size, 1))        
        self.update_network_info_group()
        # Set status
        self.status_label.setText("<b>Status:</b> Model updated")
    
    def update_network_info(self, num_layers, units_per_layer, optimizer, learning_rate):
        """ Update the network information in the UI. """
        global model
        if model is None:
            return

        # Clear existing widgets
        while self.network_info_layout.count():
            item = self.network_info_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        # Update network type and other labels
        self.network_type_label = QLabel(f"<b>Network Type:</b> LSTM - {model.__class__.__name__}")
        self.network_info_layout.addWidget(self.network_type_label)

        self.network_params_label = QLabel(f"<b>Total Parameters:</b> <code>{model.count_params()}</code>")
        self.network_info_layout.addWidget(self.network_params_label)

        self.optimizer_info = QLabel(f"<b>Optimizer:</b> <code>{optimizer}</code>")
        self.loss_info = QLabel("<b>Loss Function:</b> <code>binary_crossentropy</code>")
        self.network_info_layout.addWidget(self.optimizer_info)
        self.network_info_layout.addWidget(self.loss_info)

        # Layer info
        layers_info = QLabel(f"<b>Layers:</b>")
        self.network_info_layout.addWidget(layers_info)
        for i in range(num_layers):
            layer_info_text = f"-- Layer {i + 1}: <code>LSTM</code>, Units: <code>{units_per_layer}</code>, Activation: <code>tanh</code>"
            layer_info = QLabel(layer_info_text)
            self.network_info_layout.addWidget(layer_info)

        # Add Visualize Model button
        self.visualize_model_button = QPushButton('Show Model Architecture', self)
        # [Rest of the code for this button]
        self.network_info_layout.addWidget(self.visualize_model_button)
        
    def update_network_info_group(self):
        """ Update the network information group in the UI based on the current model. """
        global model
        num_layers = len(model.layers)  # Assuming each layer in the model should be counted
        units_per_layer = model.layers[0].units if hasattr(model.layers[0], 'units') else 'N/A'
        optimizer = model.optimizer.__class__.__name__
        learning_rate = 'N/A'  # Add logic to retrieve learning rate if available
        self.update_network_info(num_layers, units_per_layer, optimizer, learning_rate)

    def update_network_info_display(self):
        """ Update the network information display based on user inputs. """
        num_layers = self.num_layers_input.value()
        units_per_layer = self.units_per_layer_input.value()
        optimizer = self.optimizer_input.currentText()
        learning_rate = self.learning_rate_input.text()
        self.update_network_info(num_layers, units_per_layer, optimizer, learning_rate)
    
    def create_hyperparameter_input_group(self):
        hyper_group = QGroupBox("Hyperparameters")
        # Center the group box title
        hyper_group.setAlignment(Qt.AlignCenter)
        hyper_layout = QFormLayout()
        
        # Input field for the number of epochs
        self.epoch_input = QSpinBox(self)
        self.epoch_input.setMinimum(1)
        self.epoch_input.setMaximum(1000)
        self.epoch_input.setValue(10)
        hyper_layout.addRow("<b>Number of</b> Training <b>Epochs:</b>", self.epoch_input)
        
        # Number of Layers
        self.num_layers_input = QSpinBox(self)
        self.num_layers_input.setMinimum(1)
        self.num_layers_input.setMaximum(9)
        self.num_layers_input.valueChanged.connect(self.update_network_info_display)
        hyper_layout.addRow("<b>Number of</b> LSTM <b>Layers:</b>", self.num_layers_input)

        # Units Per Layer
        self.units_per_layer_input = QSpinBox(self)
        self.units_per_layer_input.setMinimum(1)
        self.units_per_layer_input.setMaximum(100)
        self.units_per_layer_input.setValue(3)
        self.units_per_layer_input.valueChanged.connect(self.update_network_info_display)
        hyper_layout.addRow("<b>Units Per Layer:</b>", self.units_per_layer_input)

        # Learning Rate
        self.learning_rate_input = QLineEdit(self)
        self.learning_rate_input.setValidator(QDoubleValidator(0.00001, 1.0, 5))
        # Set default value
        self.learning_rate_input.setText('0.1')
        hyper_layout.addRow("<b>Learning Rate:</b>", self.learning_rate_input)

        # Optimizer Selection
        self.optimizer_input = QComboBox(self)
        self.optimizer_input.addItems(['adam', 'adagrad', 'sgd'])
        self.optimizer_input.currentIndexChanged.connect(self.update_network_info_display)
        hyper_layout.addRow("<b>Optimizer:</b>", self.optimizer_input)

        # Context Size
        self.context_size_input = QSpinBox(self)
        self.context_size_input.setMinimum(1)
        self.context_size_input.setMaximum(20)
        self.context_size_input.setValue(context_size)
        hyper_layout.addRow("<b>Context Size:</b>", self.context_size_input)

        # Update Parameters Button
        self.update_params_button = QPushButton('Update Parameters', self)
        self.update_params_button.setToolTip("Update the model parameters with these values")
        self.update_params_button.setStyleSheet("background-color: #AEE4F6")
        self.update_params_button.clicked.connect(self.update_model)
        hyper_layout.addRow(self.update_params_button)  

        hyper_group.setLayout(hyper_layout)
        return hyper_group

    def create_input_group(self):
        """ Create the input group for the GUI """
        input_group = QGroupBox("Input Data")
        
        input_layout = QFormLayout()

        # Input field for the sequence
        self.input_field = QLineEdit(self)
        self.input_field.textChanged.connect(self.update_original_sequence)
        self.input_field.textChanged.connect(self.check_input_sequence)
        self.input_field.setPlaceholderText("0,1,0,1")

        input_layout.addRow("<b>Enter Sequence</b> (comma-separated, e.g., <code>0,1,0,1</code>):", self.input_field)
        input_group.setLayout(input_layout)
        
        return input_group

    def check_input_sequence(self):
        if not self.input_field.text().strip():
            self.status_label.setText("<b>Status:</b> Please enter a sequence.")
        else:
            self.status_label.setText("<b>Status:</b> Entering sequence (training data)...")
            
    def create_control_buttons(self):
        """ Create the control buttons for the GUI """
        control_group = QGroupBox("")
        main_horizontal_layout = QHBoxLayout()  # Main horizontal layout for buttons and training info

        # Vertical layout for buttons
        button_group = QGroupBox("Training Control Buttons")
        # center the group box title
        button_group.setAlignment(Qt.AlignCenter)
        
        button_layout = QVBoxLayout()
        
        # First row of buttons
        first_row_layout = QHBoxLayout()
        # Create the start button bold
        self.start_button = QPushButton('Start')
        self.start_button.setFont(QFont('Arial', 8, weight=QFont.Bold))
        self.start_button.setStyleSheet("background-color: #CADEDB")
        self.start_button.setToolTip("Start the training process")
        self.start_button.clicked.connect(self.start_training)
        first_row_layout.addWidget(self.start_button)

        self.reset_button = QPushButton('Reset')
        self.reset_button.setToolTip("Reset the training and clear all data")
        self.reset_button.setFont(QFont('Arial', 8, weight=QFont.Bold))
        self.reset_button.setStyleSheet("background-color: #F1D7D3")
        self.reset_button.clicked.connect(self.reset_training)
        first_row_layout.addWidget(self.reset_button)
        button_layout.addLayout(first_row_layout)
        
        # Second row of buttons
        second_row_layout = QHBoxLayout()

        self.stop_button = QPushButton('Pause')
        self.stop_button.setToolTip("Pause the ongoing training")
        self.stop_button.setStyleSheet("background-color: #FCF4D3")
        self.stop_button.clicked.connect(self.stop_training)
        second_row_layout.addWidget(self.stop_button)
        
        self.continue_button = QPushButton('Continue')
        self.continue_button.setToolTip("Continue the paused training")
        self.continue_button.setStyleSheet("background-color: #FCF4D3")
        self.continue_button.clicked.connect(self.continue_training)
        second_row_layout.addWidget(self.continue_button)
        button_layout.addLayout(second_row_layout)
        
        # add horizontal line separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        button_layout.addWidget(line)
        
        # Third row of buttons
        # Horizontal layout for Show Progress Button
        show_progress_layout = QHBoxLayout()

        # Show Progress Button
        self.progress_button = QPushButton('Show Training Progress')
        self.progress_button.setStyleSheet("background-color: #CADEDB")
        self.progress_button.clicked.connect(self.show_progress)
        show_progress_layout.addWidget(self.progress_button)
        button_layout.addLayout(show_progress_layout)

        button_group.setLayout(button_layout)  # Set the layout to button_group

        # Add button group to the left side of the main horizontal layout
        main_horizontal_layout.addWidget(button_group)

        # Vertical layout for training info
        training_info_layout = QVBoxLayout()
        training_info_group = self.create_training_info_group()
        training_info_layout.addWidget(training_info_group)

        # Add training info layout to the right side of the main horizontal layout
        main_horizontal_layout.addLayout(training_info_layout)

        control_group.setLayout(main_horizontal_layout)
        return control_group

    def create_output_group(self):
        """ Create the output group for the GUI """
        
        output_group = QGroupBox("")
        
        # Main horizontal layout
        main_layout = QHBoxLayout()

        # Vertical layout for labels
        output_layout = QVBoxLayout()
        self.original_sequence_label = QLabel("<b>Original</b> Sequence: N/A")
        output_layout.addWidget(self.original_sequence_label)
        self.updated_sequence_label = QLabel("<b>Updated</b> Sequence: N/A")
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
        self.predict_button.setToolTip("Make a prediction based on the input sequence")
        self.predict_button.setFont(QFont('Arial', 10, weight=QFont.Bold))
        self.predict_button.setStyleSheet("background-color: #AEE4F6")
        self.predict_button.clicked.connect(self.predict_sequence)
        self.predict_button.setMinimumHeight(50)
        self.predict_button.setMinimumWidth(100)
                
        # Add the button to the main horizontal layout
        main_layout.addWidget(self.predict_button)
        # Set the main layout for the output group
        output_group.setLayout(main_layout)
        
        return output_group

    def create_network_info_group(self):
        """ Create the network information group for the GUI """
        global model

        network_info_group = QGroupBox("Network Information")
        # Center the group box title
        network_info_group.setAlignment(Qt.AlignCenter)
        self.network_info_layout = QVBoxLayout()  # Set this as the primary layout for the group

        # Add the network type
        self.network_type_label = QLabel(f"<b>Network Type:</b> LSTM - {model.__class__.__name__}")
        self.network_info_layout.addWidget(self.network_type_label)
        
        # Add the total number of parameters
        self.network_params_label = QLabel(f"<b>Total Parameters:</b> <code>{model.count_params()}</code>")
        self.network_info_layout.addWidget(self.network_params_label)

        self.optimizer_info = QLabel(f"<b>Optimizer:</b> <code>{model.optimizer.__class__.__name__}</code>")
        self.loss_info = QLabel("<b>Loss Function:</b> <code>binary_crossentropy</code>")
        self.network_info_layout.addWidget(self.optimizer_info)
        self.network_info_layout.addWidget(self.loss_info)

        layers_info = QLabel(f"<b>Layers:</b>")
        self.network_info_layout.addWidget(layers_info)
        # Add the number of layers
        self.layer_info_labels = []
        for i, layer in enumerate(model.layers):
            layer_info = QLabel(f"-- Layer {i+1}: <code>{type(layer).__name__}</code>, "
                                f"Units: <code>{layer.units if hasattr(layer, 'units') else 'N/A'}</code>, "
                                f"Activation: <code>{layer.activation.__name__ if hasattr(layer, 'activation') else 'N/A'}</code>")
            self.network_info_layout.addWidget(layer_info)
            self.layer_info_labels.append(layer_info)

        # Button to visualize the model
        self.visualize_model_button = QPushButton('Show Model Architecture', self)
        self.visualize_model_button.setStyleSheet("background-color: #CADEDB")
        self.visualize_model_button.clicked.connect(self.visualize_model)
        self.visualize_model_button.setToolTip("Click to view the structure of the LSTM model")
        self.network_info_layout.addWidget(self.visualize_model_button)

        network_info_group.setLayout(self.network_info_layout)  # Set the layout to the group

        return network_info_group

    def create_training_info_group(self):
        """ Create the training information group for the GUI """
        
        training_info_group = QGroupBox("Training Information")
        # Center the group box title
        training_info_group.setAlignment(Qt.AlignCenter)
        training_info_layout = QVBoxLayout()

        self.train_status_label = QLabel("<b>Status:</b> Model untrained")
        training_info_layout.addWidget(self.train_status_label)
        
        # Progress Bar
        self.training_progress_bar = QProgressBar()
        #self.training_progress_bar.setMinimumWidth(300)
        self.training_progress_bar.setValue(0)  # Initially set to 0%
        training_info_layout.addWidget(self.training_progress_bar)
        
        # Add horizontal line separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        training_info_layout.addWidget(line)

        self.epoch_label = QLabel("<b>Epoch:</b> N/A")
        training_info_layout.addWidget(self.epoch_label)
        
        self.loss_label = QLabel("<b>Loss:</b> N/A")
        training_info_layout.addWidget(self.loss_label)
        
        self.accuracy_label = QLabel("<b>Accuracy:</b> N/A")
        training_info_layout.addWidget(self.accuracy_label)
        
        training_info_group.setLayout(training_info_layout)
        return training_info_group

    def update_training_info(self, info):
        """ Update the training information labels """
        self.loss_label.setText(f"<b>Loss:</b> <code>{info.get('loss', 'N/A')}</code>")
        self.accuracy_label.setText(f"<b>Accuracy:</b> <code>{info.get('accuracy', 'N/A')}</code>")
        self.epoch_label.setText(f"<b>Epoch:</b> <code>{info.get('epoch', 'N/A')}/{self.epoch_input.value()}</code>")
        self.status_label.setText(f"<b>Status:</b> Training (Epoch {info.get('epoch', 'N/A')}/{self.epoch_input.value()})")

    def update_training_progress(self, value):
        """ Update the training progress bar """
        self.training_progress_bar.setValue(value)
        
    def update_progress(self, value):
        """ Update the progress bar """
        if hasattr(self, 'progress_window'):
            self.progress_window.update_progress(value)
            
    def reset_training(self):
        """ Reset the training """
        global model, global_training_history
        self.is_model_trained = False
        self.current_epoch = 0
        model = create_model(num_layers=self.num_layers_input.value(), 
                             units_per_layer=self.units_per_layer_input.value(), 
                             learning_rate=float(self.learning_rate_input.text()), 
                             optimizer=self.optimizer_input.currentText(), 
                             loss='binary_crossentropy')
        
        model.build(input_shape=(1, self.context_size, 1))
        self.training_event.set()
        self.clear_plots()
        self.training_progress_bar.setValue(0)
        self.progress_window.update_progress(0)

        # Clear training information
        global_training_history = {'loss': [], 'accuracy': []}
        self.train_status_label.setText("<b>Status:</b> Model untrained")
        self.epoch_label.setText("<b>Epoch:</b> N/A")
        self.loss_label.setText("Loss: N/A")
        self.accuracy_label.setText("Accuracy: N/A")
        self.status_label.setText("<b>Status:</b> Ready (training reset)")
        
        # Reset training data
        self.X, self.y = None, None

    def clear_plots(self):
        """ Clear the plots """
        if hasattr(self, 'progress_window'):
            self.progress_window.clear_plots()
    
    def visualize_model(self):
        """ Visualize the model: show the architecture"""
        # Save the model plot to a temporary file
        plot_file = './model_plot.png'
        plot_model(model, to_file=plot_file, show_shapes=True, show_layer_names=True)

        # Create a widget to show the image
        image_dialog = QDialog(self)
        image_dialog.setWindowTitle("Model Visualization")
        image_dialog.setWindowModality(Qt.NonModal)  # Set the dialog as non-modal

        layout = QVBoxLayout()
        image_label = QLabel()
        pixmap = QPixmap(plot_file)
        image_label.setPixmap(pixmap)

        # Set "What's This?" text for the image label
        image_label.setWhatsThis("This image shows the architecture of the LSTM model used in the application.")
        layout.addWidget(image_label)

        image_dialog.setLayout(layout)
        image_dialog.show()  # Use show() instead of exec_() for modeless dialog
        
    def update_original_sequence(self, text):
        """ Update the original sequence label when the input field is changed"""
        if text:
            self.original_sequence_label.setText(f"Original Sequence: <code>{text}<code>")
        else:
            self.original_sequence_label.setText("Original Sequence: N/A")
    
    def predict_sequence(self):
        """ Predict the next value in the sequence when the predict button is clicked """
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
                self.train_status_label.setText("<b>Status:</b> Predicting...")
                self.status_label.setText("<b>Status:</b> Predicting...")
                self.original_sequence_label.setText(f"Original Sequence: <code>{input_sequence}</code>")
                updated_sequence = input_sequence + ',' + str(prediction)
                self.updated_sequence_label.setText(f"Updated Sequence: <code>{updated_sequence}</code>")
                # Format confidence as a percentage
                confidence_percent = f"{confidence * 100:.2f}%"
                self.prediction_label.setText(f"Predicted value for the next step: <code>{prediction}</code> (Confidence: <code>{confidence_percent}</code>)")
                self.status_label.setText("<b>Status:</b> Prediction done")
        except ValueError as e:
            QMessageBox.warning(self, 'Invalid Input', str(e))
            self.original_sequence_label.setText("Original Sequence: N/A")
            self.updated_sequence_label.setText("Updated Sequence: N/A")

    def calculate_prediction(self, input_sequence):
        """ Calculate the prediction for the next value in the sequence """
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
        """ Start the training when the start button is clicked """
        if self.training_thread is None or not self.training_thread.is_alive():
            input_sequence = self.input_field.text()
            if not input_sequence:
                QMessageBox.warning(self, 'Invalid Input', 'Please enter a sequence.')
                return

            # Retrieve hyperparameter values from the input fields
            num_layers = self.num_layers_input.value()
            units_per_layer = self.units_per_layer_input.value()
            learning_rate = float(self.learning_rate_input.text())
            optimizer = self.optimizer_input.currentText()
            context_size = self.context_size_input.value()

            # Recreate the model with the new hyperparameters
            global model
            model = create_model(num_layers=num_layers, 
                                units_per_layer=units_per_layer, 
                                learning_rate=learning_rate, 
                                optimizer=optimizer, 
                                loss='binary_crossentropy')
            model.build(input_shape=(1, context_size, 1))

            # Prepare training data
            self.X, self.y = generate_training_data(input_sequence, context_size)

            # Clear previous plots and reset progress bar
            self.clear_plots()
            self.training_event.clear()
            self.training_progress_bar.setValue(0)

            # Start the training thread
            self.training_thread = Thread(target=lambda: self.train_and_visualize(epochs=self.epoch_input.value(), batch_size=1))
            self.training_thread.start()
            self.train_status_label.setText("<b>Status:</b> training...")
            self.status_label.setText("<b>Status:</b> Training...")

    def stop_training(self):
        """ Stop the training when the stop button is clicked """
        self.training_event.set()
        self.status_label.setText("<b>Status:</b> Training paused")
        self.train_status_label.setText("<b>Status:</b> training paused")

    def continue_training(self):
        """ Continue the training when the continue button is clicked """
        if self.training_thread is not None and not self.training_thread.is_alive():
            self.training_event.clear()
            self.training_thread = Thread(target=lambda: self.train_and_visualize(epochs=self.epoch_input.value(), batch_size=1, continue_training=True))
            self.training_thread.start()
            self.status_label.setText("<b>Status:</b> Training...")
            self.train_status_label.setText("<b>Status:</b> training continued")
            # continue progress bar from the previous value
            self.training_progress_bar.setValue(self.training_progress_bar.value())
            
    def show_progress(self):
        """ Show the progress window when the show progress button is clicked """
        global global_training_history
        self.progress_window.update_plots(None, global_training_history, self.y)
        self.progress_window.show(self.y)

    def train_and_visualize(self, epochs=10, batch_size=1, continue_training=False):
        """ Train the model and visualize the training progress """
        global model, global_training_history
        history = global_training_history if continue_training else {'loss': [], 'accuracy': []}
        start_epoch = self.current_epoch if continue_training else 0
        training_paused = False

        for epoch in range(start_epoch, epochs):
            if self.training_event.is_set():
                training_paused = True
                self.status_label.setText("<b>Status:</b> Training paused")
                break

            # Perform model fitting
            h = model.fit(self.X, self.y, epochs=1, batch_size=batch_size, verbose=2)
            predictions = model.predict(self.X, batch_size=batch_size).flatten()

            # Update the training history
            history['loss'].append(h.history['loss'][0])
            history['accuracy'].append(h.history['accuracy'][0])

            self.progress_window.update_plots(predictions, history, self.y)
            
            # Calculate progress and emit signals for GUI update
            progress = int((epoch + 1) / epochs * 100)
            self.update_progress_signal.emit(progress)
            self.update_info_signal.emit({
                'loss': round(h.history['loss'][0], 4),
                'accuracy': round(h.history['accuracy'][0], 4),
                'epoch': epoch + 1
            })

            self.current_epoch = epoch + 1

        global_training_history = history
        if not training_paused:
            self.status_label.setText("<b>Status:</b> Training finished, ready for prediction")
            self.train_status_label.setText("<b>Status:</b> training finished")
            self.update_progress_signal.emit(100)
            self.is_model_trained = True
            
    def save_model(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Model", "model.h5",
                                                "HDF5 Files (*.h5);;All Files (*)", options=options)
        if file_name:
            try:
                global model
                model.save(file_name)
                QMessageBox.information(self, "Model Saved", "The model has been saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while saving the model: {e}")

    def load_model(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Model", "", 
                                                "HDF5 Files (*.h5);;All Files (*)", options=options)
        if file_name:
            try:
                global model
                model = load_model(file_name)
                QMessageBox.information(self, "Model Loaded", "The model has been loaded successfully.")
                self.update_network_info_group()  # Update the UI with loaded model information
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while loading the model: {e}")