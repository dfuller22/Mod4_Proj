class Timer():
    
    """Timer class designed to keep track of and save modeling runtimes. It
    will automatically find your local timezone. Methods are .stop, .start,
    .record, and .now"""
    
    def __init__(self, fmt="%m/%d/%Y - %I:%M %p", verbose=None):
        import tzlocal
        self.verbose = verbose
        self.tz = tzlocal.get_localzone()
        self.fmt = fmt
        
    def now(self):
        import datetime as dt
        return dt.datetime.now(self.tz)
    
    def start(self):
        if self.verbose:
            print(f'---- Timer started at: {self.now().strftime(self.fmt)} ----')
        self.started = self.now()
        
    def stop(self):
        print(f'---- Timer stopped at: {self.now().strftime(self.fmt)} ----')
        self.stopped = self.now()
        self.time_elasped = (self.stopped - self.started)
        print(f'---- Time elasped: {self.time_elasped} ----')
        
    def record(self):
        try:
            self.lap = self.time_elasped
            return self.lap
        except:
            return print('---- Timer has not been stopped yet... ----')
        
    def __repr__(self):
        return f'---- Timer object: TZ = {self.tz} ----'
    
def make_model(layers_=None, units_=[20, 10, 3], acts_=[None, 'relu', 'softmax'], loss_='categorical_crossentropy', opt='adam', mets=['accuracy'], cust_emb=None, dropout=0.3, kern_reg=None, bidirect_1=False, n_words=20000, embed_size=128):
    
    from keras.layers import LSTM, Dense, Embedding, Dropout, GlobalMaxPool1D, Bidirectional
    from keras.models import Sequential
    
    ## Instantiate model + select embedding size
    model = Sequential()
    embedding_size = embed_size

    ## To be added as 'input_dim' for input layer
    num_words = n_words
    num_words

    if cust_emb:
        model.add(cust_emb)
    else:
        ## Input layer (input)
        e = Embedding(num_words, embedding_size)
        model.add(e)

    ## Layer safety net
    if layers_ == None:
        layers_ = [LSTM, Dense, Dense]
    
    ## Making 1st LSTM layer bidirectional
    if bidirect_1:
        model.add(Bidirectional(layers_[0](units_[0], kernel_regularizer=kern_reg, return_sequences=True)))
        model.add(GlobalMaxPool1D())
    else:
        ## 1st LSTM layer + Pooling to 1d representation
        model.add(layers_[0](units_[0], kernel_regularizer=kern_reg, return_sequences=True))
        model.add(GlobalMaxPool1D())
    
    if dropout > 0 and dropout < 1:
        ## Dropout for 1st hidden layer
        model.add(Dropout(dropout))
        ## Iterative creation of following 'inner' hidden layers
        for l, u, a in zip(layers_[1:-1], units_[1:-1], acts_[1:-1]):
            model.add(l(u, kernel_regularizer=kern_reg, activation=a))
            model.add(Dropout(dropout))
        ## Creation of output layer
        model.add(layers_.pop()(units_.pop(), kernel_regularizer=kern_reg, activation=acts_.pop()))
    
    else:
        ## Iterative creation of 'inner' hidden layers
        for l, u, a in zip(layers_[1:], units_[1:], acts_[1:]):
            model.add(l(u, kernel_regularizer=kern_reg, activation=a))

    ## Compliling model with tunable params
    model.compile(loss=loss_, optimizer=opt, metrics=mets)
    
    ## Q.C.
    print('---'*20)
    print('Model Summary:')
    print('---'*20)
    display(model.summary())
    
    return model

def plot_confusion_matrix(cm, classes, cmap=None):
    
    ## Major code components pulled from: 
    ## https://github.com/learn-co-students/dsc-visualizing-confusion-matrices-lab-online-ds-pt-100719
    
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    
    ## Build matrix just colors + color check
    if cmap == None:
        cmap = plt.cm.BuPu
    plt.imshow(cm, cmap=cmap)
    
    ## Add title and axis labels + set axis scales
    plt.title('Confusion Matrix') 
    plt.ylabel('True label (Emotion)') 
    plt.xlabel('Predicted label (Emotion)')
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    ## Add labels to each cell
    thresh = cm.max() / 2.
    
    ## Here we iterate through the confusion matrix and append labels to our visualization 
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'), horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
        
    ## Add a legend
    plt.colorbar()
    plt.show()
    
    return None

def evaluate_model(model, X_tr, X_te, y_tr, y_te, time_obj, batch=64, epch=10, val_split=0.3, callb=None, plot=True, cls_labels=None, keep=False):
    import pandas as pd
    import matplotlib.pyplot as plt
    import sklearn.metrics as metrics
    
    try:
        print('---'*20)
        print('Model Training....')
        print('---'*20)
        t = time_obj()
        t.start()       
        history = model.fit(X_tr, y_tr, batch_size=batch, epochs=epch,
                            callbacks=callb, validation_split=val_split, verbose=2)
        t.stop()
        
    except Exception as e:
        print(f'Something went wrong: {e}')
    
    if plot:
        ## Figure, axes to plot results
        fig, ax = plt.subplots(ncols=2, figsize=(12,6))

        ## Converting metrics in history object to Df for ease of plotting
        res_df = pd.DataFrame(history.history)

        ## Accuracy plot
        res_df[['val_accuracy', 'accuracy']].plot(title='Validation vs. Training Accuracy', ax=ax[0])
        ax[0].grid(color='w')
        ax[0].set_facecolor('#e8e8e8')

        ## Loss plot + reveal
        res_df[['val_loss', 'loss']].plot(title='Validation vs. Training Accuracy', ax=ax[1])
        ax[1].grid(color='w')
        ax[1].set_facecolor('#e8e8e8')
        plt.show()
        
    ## Create predictions to use with sklearn classif. report
    y_hat_te = model.predict(X_te)

    ## Coercing probabilities into class predictions 
    y_hat_te_cls = y_hat_te.argmax(axis=1)
    y_te_cls = y_te.argmax(axis=1)

    ## Confusion Matrix
    print('---'*20)
    print('Confusion Matrix:')
    print('---'*20)
    cm = metrics.confusion_matrix(y_te_cls, y_hat_te_cls, normalize='true')
    if cls_labels:
        plot_confusion_matrix(cm, cls_labels)
    else:
        labs = list(set(y_te_cls))
        plot_confusion_matrix(cm, labs)
    
    
    ## Results!
    print('---'*20)
    print('Classification Report:')
    print('---'*20)
    print(metrics.classification_report(y_te_cls, y_hat_te_cls))
    
    ## Optional save of history obj.
    if keep:
        return history
    
    return None