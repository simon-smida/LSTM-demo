from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QProgressBar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from utils import global_training_history


class ProgressWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Training Progress'
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        layout = QVBoxLayout()

        # Progress Bar
        self.progress = QProgressBar(self)
        layout.addWidget(self.progress)

        # Plots
        self.fig1 = Figure()
        self.canvas1 = FigureCanvas(self.fig1)
        self.ax1 = self.fig1.add_subplot(111)

        self.fig2 = Figure()
        self.canvas2 = FigureCanvas(self.fig2)
        self.ax2 = self.fig2.add_subplot(111)

        plot_layout = QHBoxLayout()
        plot_layout.addWidget(self.canvas1)
        plot_layout.addWidget(self.canvas2)
        layout.addLayout(plot_layout)

        self.setLayout(layout)

    def update_plots(self, predictions, history, y):
        if predictions is not None and history is not None:
            self.ax1.clear()
            self.ax1.title.set_text('Predictions vs. Actual values')
            self.ax1.plot(predictions, label='Predictions', color='blue', alpha=0.5)
            self.ax1.scatter(range(len(y)), y, label='Actual', color='orange', alpha=0.5)
            self.ax1.legend()
            self.canvas1.draw()

            self.ax2.clear()
            self.ax2.title.set_text('Loss curve and Accuracy')  
            self.ax2.plot(history['loss'], label='Loss')
            self.ax2.plot(history['accuracy'], label='Accuracy')
            self.ax2.legend()
            self.canvas2.draw()

    def show(self, y):
        global global_training_history
        self.update_plots(None, global_training_history, y)
        super().show()

    def update_progress(self, value):
        self.progress.setValue(value)
        
    def clear_plots(self):
        self.ax1.clear()
        self.ax2.clear()
        self.canvas1.draw()
        self.canvas2.draw()