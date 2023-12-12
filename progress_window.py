from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QProgressBar
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from utils import global_training_history

class ProgressWindow(QWidget):
    """
    A class for creating a window to display the training progress, 
    including a progress bar and plots for visualizing predictions and training metrics.
    """

    def __init__(self):
        """ Initialize the ProgressWindow. """
        super().__init__()
        self.title = 'Training Progress'
        self.initUI()

    def initUI(self):
        """
        Set up the user interface for the progress window.
        This includes a progress bar and two plots.
        """
        self.setWindowTitle(self.title)
        layout = QVBoxLayout()

        # Initialize and add a progress bar to the layout
        self.progress = QProgressBar(self)
        layout.addWidget(self.progress)

        # Create and set up two matplotlib figures for plotting
        # First figure for predictions vs actual values
        self.fig1 = Figure()
        self.canvas1 = FigureCanvas(self.fig1)
        self.ax1 = self.fig1.add_subplot(111)

        # Second figure for loss curve and accuracy
        self.fig2 = Figure()
        self.canvas2 = FigureCanvas(self.fig2)
        self.ax2 = self.fig2.add_subplot(111)

        # Arrange the plots in a horizontal layout
        plot_layout = QHBoxLayout()
        plot_layout.addWidget(self.canvas1)
        plot_layout.addWidget(self.canvas2)
        layout.addLayout(plot_layout)

        # Set the main layout of the window
        self.setLayout(layout)

    def update_plots(self, predictions, history, y):
        """
        Update the plots with new data.

        :param predictions: Predicted values from the model.
        :param history: Dictionary containing training history (loss and accuracy).
        :param y: Actual target values for comparison.
        """
        if predictions is not None and history is not None:
            # Update the first plot with predictions vs actual values
            self.ax1.clear()
            self.ax1.title.set_text('Predictions vs. Actual values')
            self.ax1.plot(predictions, label='Predictions', color='blue', alpha=0.5)
            self.ax1.scatter(range(len(y)), y, label='Actual', color='orange', alpha=0.5)
            self.ax1.legend()
            self.canvas1.draw()

            # Update the second plot with loss and accuracy curves
            self.ax2.clear()
            self.ax2.title.set_text('Loss curve and Accuracy')  
            self.ax2.plot(history['loss'], label='Loss')
            self.ax2.plot(history['accuracy'], label='Accuracy')
            self.ax2.legend()
            self.canvas2.draw()

    def show(self, y):
        """
        Show the progress window, updating the plots using the latest training history.

        :param y: Actual target values for comparison.
        """
        global global_training_history
        self.update_plots(None, global_training_history, y)
        super().show()

    def update_progress(self, value):
        """
        Update the progress bar.

        :param value: The new value to set on the progress bar (0-100).
        """
        self.progress.setValue(value)
        
    def clear_plots(self):
        """ Clear the data from the plots. """
        self.ax1.clear()
        self.ax2.clear()
        self.canvas1.draw()
        self.canvas2.draw()