from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score

class MacroF1Score:
    def __init__(self):
        self.y_true = []
        self.y_pred = []
        
    def update(self, y_true, y_pred):
        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)
    
    def result(self):
        if not self.y_true or not self.y_pred:
            return 0.0
        print("="*50)
        print("Accuracy:", accuracy_score(self.y_true, self.y_pred))
        print("Precision:", precision_score(self.y_true, self.y_pred, average='macro', zero_division=0))
        print("Recall:", recall_score(self.y_true, self.y_pred, average='macro', zero_division=0))
        print("="*50)
        return f1_score(self.y_true, self.y_pred, average='macro', zero_division=0)
    
    def reset(self):
        self.y_true = []
        self.y_pred = []
        