from PyQt5.QtWidgets import QApplication
import sys
from app import App

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')   
    ex = App()
    ex.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()