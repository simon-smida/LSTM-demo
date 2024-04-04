# Standard library imports
from threading import Event, Thread

# Third-party imports
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QDoubleValidator
from PyQt5.QtWidgets import (QAction, QComboBox, QDialog, QFileDialog, QFormLayout, QFrame, 
                             QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, 
                             QMenuBar, QProgressBar, QPushButton, QSizePolicy, QSpacerItem, QSpinBox, 
                             QVBoxLayout, QWidget)

from tensorflow.keras.utils import plot_model

# Local application imports
from model import create_model
from progress_window import ProgressWindow
from utils import generate_training_data, global_training_history, model


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
        self.context_size =  2        # Context size for the LSTM model
        self.is_model_trained = False # Flag to track if the model is trained
        self.layer_info_labels = []   # Labels for layer information
        self.progress_window = ProgressWindow()  # Window to show training progress
        
        # Initialize the network info layout
        self.network_info_layout = QVBoxLayout()

        # Initialize UI components
        self.initUI()

        # Create the initial model
        self.create_initial_model()
        
        # Connect signals to slots
        self.update_progress_signal.connect(self.update_progress)
        self.update_info_signal.connect(self.update_training_info)
        self.update_progress_signal.connect(self.update_training_progress)
    
    def initUI(self):
        """
        Initialize the GUI components of the application. 
        This includes setting up input groups, status labels, control buttons, 
        and menu bars for the main window.
        """
        # Set the central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Initialize input group for user data input
        main_layout.addWidget(self.create_input_group())

        # Initialize and style the status label
        self.init_status_label()
        main_layout.addWidget(self.status_label)

        # Add output, control, and information panels
        main_layout.addWidget(self.create_output_group())
        main_layout.addWidget(self.create_control_buttons())
        main_layout.addWidget(self.create_combined_info_group())

        # Initialize the menu bar
        self.init_menu_bar()

        # Adjust window size to fit the layout
        min_size = main_layout.minimumSize()
        self.resize(min_size.width(), min_size.height())

    def init_status_label(self):
        """ Initialize and style the status label. """
        self.status_label = QLabel("<b>Status:</b> Ready (model untrained)")
        self.status_label.setStyleSheet("padding-top: 20px; padding-bottom: 3px;")
        self.status_label.setFont(QFont('Arial', 12))
        self.status_label.setAlignment(Qt.AlignCenter)

    def init_menu_bar(self):
        """ Initialize the menu bar with file and help menus. """
        self.create_file_menu()
        self.create_help_menu()

    def create_file_menu(self):
        """ Create the File menu with model save/load and exit actions. """
        file_menu = self.menuBar().addMenu("&Model")

        # Save Model action
        save_action = self.create_action("&Save Model", "Ctrl+S", self.save_model)
        file_menu.addAction(save_action)

        # Load Model action
        load_action = self.create_action("&Load Model", "Ctrl+O", self.load_model)
        file_menu.addAction(load_action)

        # Exit action
        exit_action = self.create_action("&Exit", "Ctrl+Q", self.close)
        file_menu.addAction(exit_action)

    def create_help_menu(self):
        """ Create the Help menu with an about action. """
        help_menu = self.menuBar().addMenu("&Help")
        about_action = self.create_action("&About", triggered=self.about_dialog)
        help_menu.addAction(about_action)

    def create_action(self, text, shortcut=None, triggered=None):
        """ Helper function to create a QAction. """
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        if triggered:
            action.triggered.connect(triggered)
        return action

    def about_dialog(self):
        """ Display the 'About' dialog box with application information. """
        QMessageBox.about(self, "About LSTM Visualization Tool",
                        "LSTM Visualization Tool\nVersion 1.0\nDeveloped by [Šimon Šmída]")

    # ===================================
    #       UI Component Creators
    # ===================================
    def create_input_group(self):
        """
        Create the input group for the GUI.
        This group allows the user to enter a sequence for processing.
        """
        input_group = QGroupBox("Input Data")
        input_layout = QFormLayout()

        self.input_field = QLineEdit(self)
        self.input_field.textChanged.connect(self.update_original_sequence)
        self.input_field.textChanged.connect(self.on_input_change)
        self.input_field.textChanged.connect(self.check_input_sequence)
        self.input_field.setPlaceholderText("0,1,0,1")
        input_layout.addRow("<b>Enter Sequence</b> (comma-separated, e.g., 0,1,0,1):", self.input_field)

        input_group.setLayout(input_layout)
        return input_group

    def create_output_group(self):
        """
        Create the output group for the GUI.
        This group displays the original and updated sequences, and the prediction result.
        """
        output_group = QGroupBox("Output Information")
        main_layout = QHBoxLayout()

        output_layout = QVBoxLayout()
        self.original_sequence_label = QLabel("<b>Original Sequence:</b> N/A")
        self.updated_sequence_label = QLabel("<b>Updated Sequence:</b> N/A")
        self.prediction_label = QLabel("Prediction will appear here after processing.")
        output_layout.addWidget(self.original_sequence_label)
        output_layout.addWidget(self.updated_sequence_label)
        output_layout.addWidget(self.prediction_label)

        main_layout.addLayout(output_layout)
        main_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.predict_button = self.create_predict_button()
        main_layout.addWidget(self.predict_button)
        output_group.setLayout(main_layout)

        return output_group

    def create_predict_button(self):
        """ Create the 'Predict' button. """
        predict_button = QPushButton('Predict', self)
        predict_button.setToolTip("Make a prediction based on the input sequence")
        predict_button.setFont(QFont('Arial', 10, weight=QFont.Bold))
        predict_button.setStyleSheet("background-color: #AEE4F6")
        predict_button.clicked.connect(self.predict_sequence)
        predict_button.setMinimumHeight(50)
        predict_button.setMinimumWidth(100)
        return predict_button

    # ===================================
    def create_control_buttons(self):
        """
        Create the control buttons for the GUI.
        Includes buttons for starting, pausing, continuing, and resetting training.
        """
        control_group = QGroupBox("Training Control")
        main_horizontal_layout = QHBoxLayout(control_group)

        button_group = QGroupBox("Control Buttons")
        button_group.setAlignment(Qt.AlignCenter)
        button_layout = QVBoxLayout()

        # Add training control buttons
        training_control_layout = self.create_training_control_buttons()
        button_layout.addLayout(training_control_layout)

        # Add horizontal line
        line = self.create_horizontal_line()
        button_layout.addWidget(line)

        # Add progress control button
        progress_button_layout = self.create_progress_control_buttons()
        button_layout.addLayout(progress_button_layout)

        button_group.setLayout(button_layout)
        main_horizontal_layout.addWidget(button_group)

        training_info_group = self.create_training_info_group()
        main_horizontal_layout.addWidget(training_info_group)

        return control_group

    def create_training_control_buttons(self):
        """ Create buttons for training control: Start, Reset, Pause, Continue. """
        layout = QVBoxLayout()

        # First row: Start and Reset buttons
        first_row_layout = QHBoxLayout()
        self.start_button = self.create_button('Start', 'Start the training process', 'background-color: #CADEDB', self.start_training)
        self.reset_button = self.create_button('Reset', 'Reset the training process', 'background-color: #F1D7D3', self.reset_training)
        first_row_layout.addWidget(self.start_button)
        first_row_layout.addWidget(self.reset_button)

        # Second row: Pause and Continue buttons
        second_row_layout = QHBoxLayout()
        self.pause_button = self.create_button('Pause', 'Pause the training process', 'background-color: #FCF4D3', self.stop_training)
        self.continue_button = self.create_button('Continue', 'Continue the training process', 'background-color: #FCF4D3', self.continue_training)
        second_row_layout.addWidget(self.pause_button)
        second_row_layout.addWidget(self.continue_button)

        layout.addLayout(first_row_layout)
        layout.addLayout(second_row_layout)
        return layout

    def create_progress_control_buttons(self):
        """ Create a button for showing training progress. """
        layout = QHBoxLayout()
        self.progress_button = self.create_button('Show Training Progress', 'Show training progress', 'background-color: #CADEDB', self.show_progress)
        layout.addWidget(self.progress_button)
        return layout

    def create_horizontal_line(self):
        """ Helper method to create a horizontal line separator. """
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    # ===================================
    def create_hyperparameter_input_group(self):
        """
        Create the hyperparameter input group for the GUI.
        This group allows the user to set various parameters for model training.
        """
        hyper_group = QGroupBox("Hyperparameters")
        hyper_group.setAlignment(Qt.AlignCenter)
        hyper_layout = QFormLayout()

        # Epoch input
        self.epoch_input = self.create_spin_box(1, 1000, 10, "Number of training epochs")
        hyper_layout.addRow("<b>Number of Epochs:</b>", self.epoch_input)

        # Number of Layers input
        self.num_layers_input = self.create_spin_box(1, 9, 1, "Number of LSTM layers")
        self.num_layers_input.valueChanged.connect(self.update_network_info)
        hyper_layout.addRow("<b>Number of LSTM Layers:</b>", self.num_layers_input)

        # Units Per Layer input
        self.units_per_layer_input = self.create_spin_box(1, 100, 3, "Number of units per LSTM layer")
        self.units_per_layer_input.valueChanged.connect(self.update_network_info)
        hyper_layout.addRow("<b>Units Per Layer:</b>", self.units_per_layer_input)

        # Learning Rate input
        self.learning_rate_input = self.create_line_edit('0.1', "Learning rate for the optimizer")
        hyper_layout.addRow("<b>Learning Rate:</b>", self.learning_rate_input)

        # Optimizer selection
        self.optimizer_input = self.create_combo_box(['adam', 'adagrad', 'sgd'], "Choice of optimizer for training the model")
        self.optimizer_input.currentIndexChanged.connect(self.update_network_info)
        hyper_layout.addRow("<b>Optimizer:</b>", self.optimizer_input)

        # Context Size input
        self.context_size_input = self.create_spin_box(1, 20, self.context_size, "Input context size for the LSTM model")
        self.context_size = self.context_size_input.value()
        self.context_size_input.valueChanged.connect(lambda: setattr(self, 'context_size', self.context_size_input.value()))
        hyper_layout.addRow("<b>Context Size:</b>", self.context_size_input)

        # Update Parameters button
        self.update_params_button = self.create_button("Update Parameters", "Update the model parameters", "background-color: #AEE4F6", self.update_model)
        hyper_layout.addRow(self.update_params_button)

        hyper_group.setLayout(hyper_layout)
        return hyper_group

    def create_spin_box(self, min_val, max_val, default_val, tooltip):
        """ Helper method to create a QSpinBox with specified properties. """
        spin_box = QSpinBox(self)
        spin_box.setMinimum(min_val)
        spin_box.setMaximum(max_val)
        spin_box.setValue(default_val)
        spin_box.setToolTip(tooltip)
        return spin_box

    def create_line_edit(self, default_text, tooltip):
        """ Helper method to create a QLineEdit with specified properties. """
        line_edit = QLineEdit(self)
        line_edit.setValidator(QDoubleValidator(0.00001, 1.0, 5))
        line_edit.setText(default_text)
        line_edit.setToolTip(tooltip)
        return line_edit

    def create_combo_box(self, items, tooltip):
        """ Helper method to create a QComboBox with specified items. """
        combo_box = QComboBox(self)
        combo_box.addItems(items)
        combo_box.setToolTip(tooltip)
        return combo_box

    def create_button(self, text, tooltip, style, on_click):
        """ Helper method to create a styled QPushButton. """
        button = QPushButton(text, self)
        button.setToolTip(tooltip)
        button.setStyleSheet(style)
        button.clicked.connect(on_click)
        return button
    # ===================================

    def create_network_info_group(self):
        """ Create the network information group for the GUI. """
        network_info_group = QGroupBox("Network Information")
        network_info_group.setAlignment(Qt.AlignCenter)
        self.network_info_layout = QVBoxLayout()

        self.add_model_type_label()
        self.add_model_params_label()
        self.add_optimizer_info_label()
        self.add_loss_info_label()

        # Call add_layer_info_labels with the number of layers
        num_layers = len(model.layers) if model else 0
        self.add_layer_info_labels(num_layers)

        self.add_visualize_model_button()
        network_info_group.setLayout(self.network_info_layout)
        return network_info_group
    
    def add_model_type_label(self):
        # Dynamically retrieve model type if possible
        model_type = model.__class__.__name__ if model else "N/A"
        self.network_type_label = QLabel(f"<b>Network Type:</b> LSTM - {model_type}")
        self.network_info_layout.addWidget(self.network_type_label)

    def add_model_params_label(self):
        """ Add the model parameters label to the network information group. """
        model_params = model.count_params() if model else "N/A"
        self.network_params_label = QLabel(f"<b>Total Parameters:</b> <code>{model_params}</code>")
        self.network_info_layout.addWidget(self.network_params_label)
        
    def add_optimizer_info_label(self):
        """ Add the optimizer information label to the network information group. """
        optimizer = model.optimizer.__class__.__name__ if model else "N/A"
        self.optimizer_info = QLabel(f"<b>Optimizer:</b> <code>{optimizer}</code>")
        self.network_info_layout.addWidget(self.optimizer_info)
        
    def add_loss_info_label(self):
        """ Add the loss information label to the network information group. """
        self.loss_info = QLabel("<b>Loss Function:</b> <code>binary_crossentropy</code>")
        self.network_info_layout.addWidget(self.loss_info)
        
    def add_layer_info_labels(self, num_layers):
        """ Add the layer information labels to the network information group. """
        layers_info = QLabel(f"<b>Layers:</b>")
        self.network_info_layout.addWidget(layers_info)

        for i in range(num_layers):
            # Retrieve layer-specific information if available
            layer_info_text = "Layer information not available"
            if model and i < len(model.layers):
                layer = model.layers[i]
                units = layer.units if hasattr(layer, 'units') else 'N/A'
                activation = layer.activation.__name__ if hasattr(layer, 'activation') else 'N/A'
                layer_info_text = f"-- Layer {i + 1}: <code>{type(layer).__name__}</code>, Units: <code>{units}</code>, Activation: <code>{activation}</code>"

            layer_info = QLabel(layer_info_text)
            self.network_info_layout.addWidget(layer_info)
    
    def add_visualize_model_button(self):
        """ Add the 'Visualize Model' button to the network information group. """
        self.visualize_model_button = self.create_button('Show Model Architecture', 'Show the model architecture', 'background-color: #CADEDB', self.show_model_architecture)
        self.visualize_model_button.clicked.connect(self.show_model_architecture)
        self.network_info_layout.addWidget(self.visualize_model_button)

    def show_model_architecture(self):
        """ Display the model architecture in a new window. """
        if model is None:
            QMessageBox.warning(self, 'Model Untrained', 'Please train the model before visualizing.')
            return
        plot_model(model, to_file='./img/model.png', show_shapes=True)
        pixmap = QPixmap('./img/model.png')
        self.model_architecture_window = QDialog()
        self.model_architecture_window.setWindowTitle("Model Architecture")
        self.model_architecture_window.resize(pixmap.width(), pixmap.height())
        self.model_architecture_window.label = QLabel(self.model_architecture_window)
        self.model_architecture_window.label.setPixmap(pixmap)
        self.model_architecture_window.label.show()
        self.model_architecture_window.exec_()
    
    # ===================================
    def create_training_info_group(self):
        """
        Create the training information group for the GUI.
        Displays the current status of model training, including progress and metrics.
        """
        training_info_group = QGroupBox("Training Information")
        training_info_group.setAlignment(Qt.AlignCenter)
        training_info_layout = QVBoxLayout()

        self.train_status_label = QLabel("<b>Status:</b> Model untrained")
        training_info_layout.addWidget(self.train_status_label)

        self.training_progress_bar = self.create_progress_bar()
        training_info_layout.addWidget(self.training_progress_bar)

        training_info_layout.addWidget(self.create_horizontal_line())

        self.epoch_label = QLabel("<b>Epoch:</b> N/A")
        self.loss_label = QLabel("<b>Loss:</b> N/A")
        self.accuracy_label = QLabel("<b>Accuracy:</b> N/A")
        training_info_layout.addWidget(self.epoch_label)
        training_info_layout.addWidget(self.loss_label)
        training_info_layout.addWidget(self.accuracy_label)

        training_info_group.setLayout(training_info_layout)
        return training_info_group

    def create_progress_bar(self):
        """ Helper method to create a progress bar. """
        progress_bar = QProgressBar()
        progress_bar.setValue(0)
        return progress_bar

    # ===================================
    def create_combined_info_group(self):
        """
        Create a combined group box for network information and hyperparameters.
        Helps in organizing the UI layout efficiently.
        """
        combined_group = QGroupBox("")
        combined_info_layout = QHBoxLayout()
        combined_info_layout.addWidget(self.create_network_info_group())
        combined_info_layout.addWidget(self.create_hyperparameter_input_group())
        combined_group.setLayout(combined_info_layout)
        return combined_group

    # ===================================
    #       Model-Related Methods
    # ===================================
    def create_initial_model(self):
        """ Create the initial model with default parameters. """
        from tensorflow.keras.models import load_model
        
        self.status_label.setText('Loading model, please wait...')
        
        # Set default hyperparameters
        default_num_layers = 1
        default_units_per_layer = 3
        default_learning_rate = 0.1
        default_optimizer = 'adam'

        # Build the initial model
        self.build_model(default_num_layers, default_units_per_layer, 
                        default_learning_rate, default_optimizer)

        # Update UI with initial model info
        self.update_network_info()
        self.status_label.setText("<b>Status:</b> Initial model created")

    def build_model(self, num_layers, units_per_layer, learning_rate, optimizer):
        """ Build the LSTM model with the given hyperparameters. """
        global model
        model = create_model(num_layers=num_layers, 
                            units_per_layer=units_per_layer, 
                            learning_rate=learning_rate, 
                            optimizer=optimizer, 
                            loss='binary_crossentropy')
        model.build(input_shape=(1, self.context_size, 1))

    def update_model(self):
        """ Update the model based on the hyperparameter inputs. """
        try:
            num_layers = self.num_layers_input.value()
            units_per_layer = self.units_per_layer_input.value()
            learning_rate = float(self.learning_rate_input.text())
            optimizer = self.optimizer_input.currentText()

            # Build model with new parameters
            self.build_model(num_layers, units_per_layer, learning_rate, optimizer)

            # Update UI and provide feedback
            self.update_network_info()
            self.status_label.setText("<b>Status:</b> Model updated successfully")
        except ValueError as e:
            QMessageBox.warning(self, 'Invalid Input', f'Error updating model: {e}')

    # ===================================
    # Network Information Update Methods
    # ===================================
    def update_network_info(self):
        """ Update the network information in the UI based on the current model. """
        global model
        if not model:
            return

        # Clear existing widgets in the layout
        while self.network_info_layout.count():
            item = self.network_info_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # Dynamically retrieve model information
        model_type = model.__class__.__name__
        model_params = model.count_params()
        optimizer = model.optimizer.__class__.__name__ if model.optimizer else "N/A"
        loss = model.loss if isinstance(model.loss, str) else model.loss.__name__ if model.loss else "N/A"

        # Update network type and other labels
        self.network_type_label = QLabel(f"<b>Network Type:</b> LSTM - {model_type}")
        self.network_params_label = QLabel(f"<b>Total Parameters:</b> <code>{model_params}</code>")
        self.optimizer_info = QLabel(f"<b>Optimizer:</b> <code>{optimizer}</code>")
        self.loss_info = QLabel(f"<b>Loss Function:</b> <code>{loss}</code>")

        # Add labels to layout
        self.network_info_layout.addWidget(self.network_type_label)
        self.network_info_layout.addWidget(self.network_params_label)
        self.network_info_layout.addWidget(self.optimizer_info)
        self.network_info_layout.addWidget(self.loss_info)

        # Layer information
        layers_info = QLabel("<b>Layers:</b>")
        self.network_info_layout.addWidget(layers_info)
        for i, layer in enumerate(model.layers):
            layer_type = type(layer).__name__
            units = layer.units if hasattr(layer, 'units') else 'N/A'
            activation = layer.activation.__name__ if hasattr(layer, 'activation') else 'N/A'
            layer_info_text = f"-- Layer {i + 1}: <code>{layer_type}</code>, Units: <code>{units}</code>, Activation: <code>{activation}</code>"
            layer_info = QLabel(layer_info_text)
            self.network_info_layout.addWidget(layer_info)

        # Visualize Model button
        self.visualize_model_button = QPushButton('Show Model Architecture', self)
        self.visualize_model_button.setStyleSheet("background-color: #CADEDB")
        self.visualize_model_button.clicked.connect(self.visualize_model)
        self.visualize_model_button.setToolTip("Click to view the structure of the LSTM model")
        self.network_info_layout.addWidget(self.visualize_model_button)

    # ===================================
    #       Training Control Methods
    # ===================================
    def start_training(self):
        """ Start the training when the start button is clicked. """
        if self.training_thread and self.training_thread.is_alive():
            QMessageBox.information(self, 'Training in Progress', 'The model is currently being trained.')
            return

        # Validate and retrieve the input sequence
        input_sequence = self.input_field.text().strip()
        if not self.is_valid_sequence(input_sequence):
            QMessageBox.warning(self, 'Invalid Input', 'Please enter a valid sequence.')
            return

        # Retrieve and validate hyperparameters
        try:
            num_layers, units_per_layer, learning_rate, optimizer, context_size = self.retrieve_hyperparameters()
        except ValueError as e:
            QMessageBox.warning(self, 'Invalid Hyperparameters', str(e))
            return

        # Prepare the model and training data
        preparation_result = self.prepare_model_and_data(num_layers, units_per_layer, learning_rate, optimizer, context_size, input_sequence)
        if not preparation_result:
            return
        
        # Start the training thread
        self.training_thread = Thread(target=lambda: self.train_and_visualize(epochs=self.epoch_input.value(), batch_size=1))
        self.training_thread.start()
        self.train_status_label.setText("<b>Status:</b> Training started...")
        self.status_label.setText("<b>Status:</b> Training in progress...")

    def is_valid_sequence(self, sequence):
        """ Validate the input sequence. """
        # Add specific validation logic here, e.g., check format, length, etc.
        return bool(sequence)

    def retrieve_hyperparameters(self):
        """ Retrieve and validate hyperparameters from the UI inputs. """
        try:
            num_layers = self.num_layers_input.value()
            units_per_layer = self.units_per_layer_input.value()
            learning_rate = float(self.learning_rate_input.text())
            optimizer = self.optimizer_input.currentText()
            context_size = self.context_size_input.value()
            return num_layers, units_per_layer, learning_rate, optimizer, context_size
        except ValueError as e:
            raise ValueError("Error in hyperparameter inputs: " + str(e))

    def prepare_model_and_data(self, num_layers, units_per_layer, learning_rate, optimizer, context_size, input_sequence):
        """ Prepare the model and training data. """
        global model
        model = create_model(num_layers=num_layers, units_per_layer=units_per_layer, 
                            learning_rate=learning_rate, optimizer=optimizer, loss='binary_crossentropy')
        model.build(input_shape=(1, context_size, 1))

        try:
            self.X, self.y = generate_training_data(input_sequence, context_size)
        except ValueError as e:
            QMessageBox.warning(self, 'Invalid Input', str(e))
            return False  # Indicate that the preparation was not successful

        self.clear_plots()
        self.training_event.clear()
        self.training_progress_bar.setValue(0)
        return True

       
    def stop_training(self):
        """ Stop the training when the stop button is clicked. """
        if not self.training_thread or not self.training_thread.is_alive():
            QMessageBox.information(self, 'Training Not Active', 'There is no active training to pause.')
            return

        # Check if training is already paused
        if self.training_event.is_set():
            QMessageBox.information(self, 'Training Already Paused', 'The training process is already paused.')
            return

        # Pause the training process
        self.training_event.set()

        # Update the UI to reflect the training's paused status
        self.status_label.setText("<b>Status:</b> Training paused")
        self.train_status_label.setText("<b>Status:</b> Training has been paused")

    def continue_training(self):
        """ Continue the paused training when the continue button is clicked. """
        # Check if there is a training process to continue
        if not self.training_thread:
            QMessageBox.warning(self, 'No Training Process', 'There is no training process to continue.')
            return

        # Check if the training process is already active
        if self.training_thread.is_alive():
            QMessageBox.information(self, 'Training In Progress', 'The training is already in progress.')
            return

        # Check if the training was paused
        if not self.training_event.is_set():
            QMessageBox.information(self, 'Training Not Paused', 'The training process is not paused and cannot be continued.')
            return

        # Continue the paused training process
        self.training_event.clear()
        self.training_thread = Thread(target=lambda: self.train_and_visualize(epochs=self.epoch_input.value(), batch_size=1, continue_training=True))
        self.training_thread.start()

        # Update UI to reflect the resumption of training
        self.status_label.setText("<b>Status:</b> Training resumed")
        self.train_status_label.setText("<b>Status:</b> Training has been resumed")

    def reset_training(self):
        """ Reset the training and prepare for a new training session. """
        # Confirm with the user before resetting the training
        reply = QMessageBox.question(self, 'Reset Training', 
                                    'Are you sure you want to reset the training? This will clear all current progress.',
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.No:
            return

        # Reset training-related attributes
        global model, global_training_history
        self.is_model_trained = False
        self.current_epoch = 0
        self.training_event.clear()  # Ensure the event is cleared if training was paused

        # Rebuild the model with current hyperparameter settings
        try:
            self.build_model(self.num_layers_input.value(), 
                            self.units_per_layer_input.value(), 
                            float(self.learning_rate_input.text()), 
                            self.optimizer_input.currentText())
        except ValueError as e:
            QMessageBox.warning(self, 'Invalid Hyperparameters', f'Error resetting model: {e}')
            return

        # Clear training history and reset UI components
        global_training_history = {'loss': [], 'accuracy': []}
        self.clear_plots()
        self.training_progress_bar.setValue(0)

        # Update UI status messages
        self.update_training_status("Model untrained and ready for new training session.")
    
    def update_training_status(self, message):
        """ Update training-related status messages on the UI. """
        self.train_status_label.setText(f"<b>Status:</b> {message}")
        self.epoch_label.setText("<b>Epoch:</b> N/A")
        self.loss_label.setText("<b>Loss:</b> N/A")
        self.accuracy_label.setText("<b>Accuracy:</b> N/A")
        self.status_label.setText(f"<b>Status:</b> {message}")
        
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
    
    # ===================================
    #   Progress and Prediction Methods
    # ===================================
    def update_progress(self, value):
        """ Update the progress bar """
        if hasattr(self, 'progress_window'):
            self.progress_window.update_progress(value)

    def update_training_info(self, info):
        """ Update the training information labels """
        self.loss_label.setText(f"<b>Loss:</b> <code>{info.get('loss', 'N/A')}</code>")
        self.accuracy_label.setText(f"<b>Accuracy:</b> <code>{info.get('accuracy', 'N/A')}</code>")
        self.epoch_label.setText(f"<b>Epoch:</b> <code>{info.get('epoch', 'N/A')}/{self.epoch_input.value()}</code>")
        self.status_label.setText(f"<b>Status:</b> Training (Epoch {info.get('epoch', 'N/A')}/{self.epoch_input.value()})")

    def update_training_progress(self, value):
        """ Update the training progress bar """
        self.training_progress_bar.setValue(value)

    def show_progress(self):
        """ Show the progress window when the show progress button is clicked """
        global global_training_history
        self.progress_window.update_plots(None, global_training_history, self.y)
        self.progress_window.show(self.y)
    
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
                self.original_sequence_label.setText(f"<b>Original Sequence:</b> <code>{input_sequence}</code>")
                updated_sequence = input_sequence + ',' + f"<span style='color: green; font-weight: bold;'>{prediction}</span>"
                self.updated_sequence_label.setText(f"<b>Updated Sequence:</b> <code>{updated_sequence}</code>")
                # Format confidence as a percentage
                confidence_percent = f"{confidence * 100:.2f}%"
                predicted_value = f"<span style='color: green; font-weight: bold;'>{prediction}</span>"
                self.prediction_label.setText(f"<b>Predicted Value</b> for the next step: <code>{predicted_value}</code> (Confidence: <code>{confidence_percent}</code>)")
                self.status_label.setText("<b>Status:</b> Prediction done")
        except ValueError as e:
            QMessageBox.warning(self, 'Invalid Input', str(e))
            self.original_sequence_label.setText("<b>Original Sequence:</b> N/A")
            self.updated_sequence_label.setText("<b>Updated Sequence:</b> N/A")

    def calculate_prediction(self, input_sequence):
        """ Calculate the prediction for the next value in the sequence """
        try:
            data = [int(x) for x in input_sequence.split(',')]
            
            # Ensure the input sequence length is at least equal to context_size
            if len(data) < self.context_size:
                raise ValueError(f"Input sequence [{len(data)}] is too short for the context size [{self.context_size}].")

            # Use the last context_size elements for prediction
            last_n_elements = data[-self.context_size:]
            processed_input = np.array(last_n_elements).reshape(1, self.context_size, 1)
            prediction_output = model.predict(processed_input).flatten()
            prediction = int(np.round(prediction_output[-1]))

            confidence = prediction_output[-1] if prediction == 1 else 1 - prediction_output[-1]
            confidence_percentage = round(confidence, 2)
            return prediction, confidence_percentage
        except ValueError as e:
            QMessageBox.warning(self, 'Invalid Input', str(e))
            return None, None

    # ===================================
    #           Utility Methods
    # ===================================
    def is_valid_sequence(self, sequence):
        """ Check if the sequence is a valid comma-separated sequence of zeros and ones. """
        # Check if the sequence is empty
        if not sequence:
            return False

        # Split the sequence by commas and check each part
        parts = sequence.split(',')
        for part in parts:
            if part not in ['0', '1']:
                return False

        # Ensure the sequence does not end with a comma
        if sequence.endswith(','):
            return False

        return True
    
    def on_input_change(self):
        """ Called whenever the text in the input field changes. """
        input_sequence = self.input_field.text()
        if not self.is_valid_sequence(input_sequence):
            # Update the status label to indicate invalid input
            self.status_label.setText("<b>Status:</b> Invalid input. Enter a comma-separated sequence of zeros and ones.")
        else:
            # Update the status label to indicate valid input
            self.status_label.setText("<b>Status:</b> Valid sequence entered.")

    def clear_plots(self):
        """ Clear the plots """
        if hasattr(self, 'progress_window'):
            self.progress_window.clear_plots()
    
    def visualize_model(self):
        """ Visualize the model: show the architecture"""
        # Save the model plot to a temporary file
        plot_file = './img/model_plot.png'
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
            self.original_sequence_label.setText(f"<b>Original Sequence:</b> <code>{text}<code>")
            self.updated_sequence_label.setText("<b>Updated Sequence:</b> N/A")
        else:
            self.original_sequence_label.setText("<b>Original Sequence:</b> N/A")
            self.updated_sequence_label.setText("<b>Updated Sequence:</b> N/A")
    
    def check_input_sequence(self):
        """ Check the input sequence when the input field is changed """
        if not self.input_field.text().strip():
            self.status_label.setText("<b>Status:</b> Please enter a sequence.")
        else:
            self.status_label.setText("<b>Status:</b> Entering sequence (training data)...")
        
    # ===================================
    #       Model Persistence Methods
    # =================================== 
    def save_model(self):
        """ Save the model to a file when the save model action is triggered. """
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
        """ Load a model from a file when the load model action is triggered. """
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Model", "", 
                                                "HDF5 Files (*.h5);;All Files (*)", options=options)
        if file_name:
            try:
                global model
                model = load_model(file_name)
                QMessageBox.information(self, "Model Loaded", "The model has been loaded successfully.")
                self.update_network_infop()  # Update the UI with loaded model information
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while loading the model: {e}")