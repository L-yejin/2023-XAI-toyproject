from PySide6.QtWidgets import QWidget, QApplication
from background_form import Ui_Form

class BackgroundChanger(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super(BackgroundChanger, self). __init__(parent)
        self.setupUi(self)
        


if __name__ == '__main__':
    app = QApplication()
    window = BackgroundChanger()
    window.show()
    app.exec()
