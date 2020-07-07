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
    """Builds and compiles a trainable Keras Sequential model for NLP tasks. There are mulitple
       tunable parameters with respect to layout, but it is meant to create an embedding layer 
       followed by one LSTM and one output layer at minimum.

    Args:
        layers_ ([list], optional): Must be list of Keras layers. Ex: [LSTM, Dense] Defaults to None.
        units_ (list, optional): Amount of desired neurons in each layer. Defaults to [20, 10, 3].
        acts_ (list, optional): Acitivation functions to be used. First layer must be None. Defaults to [None, 'relu', 'softmax'].
        loss_ (str, optional): Desired Keras compatible loss function. Defaults to 'categorical_crossentropy'.
        opt (str, optional): Desired Keras compatible optimizer. Defaults to 'adam'.
        mets (list, optional): Desired Keras compatible metric(s). Defaults to ['accuracy'].
        cust_emb (Keras Embedding, optional): Must be pre-made Keras Embedding layer (or equivalent). 
            'None' creates trainable embeddings set to 'embed_size'. Defaults to None.
        dropout (float, optional): Total amount of dropout desired *per layer*. Set to 0 to disable. Defaults to 0.3.
        kern_reg (str, optional): Desired Keras compatible kernel regularizer. Defaults to None.
        bidirect_1 (bool, optional): Enables 1st LSTM layer to be bi-directional. Defaults to False.
        n_words (int, optional): Number of words to be used in dimensions of trainable embedding layer. Defaults to 20000.
        embed_size (int, optional): Setting dimensionality of the learning space for trainable embedding layer. Defaults to 128.

    Returns:
        keras.model: Fully compiled Keras model ready for training [.fit()]
    """

    from keras.layers import LSTM, Dense, Embedding, Dropout, GlobalAveragePooling1D, Bidirectional
    from keras.models import Sequential
    
    ## Instantiate model + select embedding size
    model = Sequential()
    embedding_size = embed_size

    ## To be added as 'input_dim' for input layer
    num_words = n_words
    num_words

    if cust_emb:
        model.add(cust_emb)
        print('\n')
        print('Using Pre-Trained Embedding!')
        print('\n')
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
        model.add(GlobalAveragePooling1D())
    else:
        ## 1st LSTM layer + Pooling to 1d representation
        model.add(layers_[0](units_[0], kernel_regularizer=kern_reg, return_sequences=True))
        model.add(GlobalAveragePooling1D())
    
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
    """Converts Sklearn confusion matrix into clean matplotlib plot. Can customize colors if desired.

    Args:
        cm (Confusion Matrix): Designed to take in output from sklearn.metrics.confusion_matrix
        classes (list): Labels (as str) to be used on axes of cm
        cmap (Colormap, optional): Must be Matplotlib compatible colormap. 'None' will select 'plt.cm.BuPu'. Defaults to None.

    Returns:
        Confustion Matrix: Matplotlib figure, axis with pretty-fied normalized confusion martix
    """

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
    """Designed to quickly train, produce and visualize results from a Keras neural network.
        Option to keep results saved in history object from .fit().

    Args:
        model (Keras Model): Must be fully compiled Keras model to work properly. 
        X_tr (iterable): Training set of features. Designed for arrays or pandas objects.
        X_te (iterable): Test set of features. Designed for arrays or pandas objects.
        y_tr (iterable): Training set for target(s). Designed for arrays or pandas objects.
        y_te (iterable): Test set for for target(s). Designed for arrays or pandas objects.
        time_obj (object): Designed for custom timer class. Copy code from this file to use.
        batch (int, optional): Size of mini-batch to be used during training. Defaults to 64.
        epch (int, optional): Number of epochs to train model. Defaults to 10.
        val_split (float, optional): Proportion of training data to use for validation during 
            training. Defaults to 0.3 (30%).
        callb (list, optional): If used, must be list of Keras compatible callbacks such as 
            EarlyStopping. Defaults to None.
        plot (bool, optional): Enables plotting of accuracy and loss curves. Defaults to True.
        cls_labels (list, optional): List of strings to be used in confusion matrix. 'None' 
            will use labels derived from 'y_te'. Defaults to None.
        keep (bool, optional): Enables output of Keras history object from trained model. Defaults to False.

    Returns:
        Keras model.history: Optional output. Otherwise will display visuals and return 'None'
    """

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
        res_df[['val_loss', 'loss']].plot(title='Validation vs. Training Loss', ax=ax[1])
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

def regex_cleaner(data, pattern, repl, keep=True):
    """Helper function meant to take in regex pattern to clean text data.
       Returns list of transformed text.

    Args:
        data (iterable): Designed with arrays and pandas objects filled with strings.
        pattern (str): Regex formatted string. Ex: r"(w\+)".
        repl (str): String object desired to replace captured tokens.
        keep (bool, optional): Enables output of transformed text. Otherwise will 
            display only. Defaults to True.

    Returns:
        list: List of transformed 'data' according to regex pattern and replacement.
    """
    import regex
    
    ## Creating a container for results
    results  = []
    
    ## Looping through ea. tweet, subbing + stuffing in container 
    for item in data:
        inb = regex.sub(pattern, repl, item)
        results.append(inb)
    
    if not keep:
        return display(results)
        
    return results

def regex_cleanse(data, patterns, repl):
    """Operationalizes 'regex_cleaner' function for use with mulitple patterns and 
       replacement string(s). Returns list of transformed text.

    Args:
        data (iterable): Designed with arrays and pandas objects filled with strings.
        patterns (iterable): List of regex formatted strings. Ex: [r"(w\+)", r"(d\+)"]
        repl (str): Can be str or list of strings corresponding to patterns for cleaning
            text data.

    Returns:
        list: List of transformed 'data according to regex patterns and replacement(s).
    """
    
    ## Check if lists match + setting baseline for further removal
    if len(repl) == len(patterns):
        res = regex_cleaner(data, patterns[0], repl[0])
        
        try:
            ## Looping through remaining patterns
            for p, r in zip(patterns[1:], repl[1:]):
                res = regex_cleaner(res, p, r)
            return res
        
        except:
            print('----Someting went wrong!----')
    
    ## For simple string replacement
    elif type(repl) is str:
        res = regex_cleaner(data, patterns[0], repl)
        
        try:
            ## Looping through remaining patterns
            for p in patterns[1:]:
                res = regex_cleaner(res, p, repl)
            return res
        
        except:
            print('----Someting went wrong!----')
            
def transform_format_img(val):
    """Helper function for use in 'word_cloud_viz'. Converts 0 values in RGB arrays
       into 255, so that the WordCloud function will process the mask's white areas
       correctly.

    Args:
        val (int): Designed to help parse of nparray respresentation of .png image.

    Returns:
        int: 255 from any values of 0.
    """

    if val == 0:
        return 255
    else:
        return val
    
def word_cloud_viz(wc_list, mask_path='Twitter_logo.png', save_path='img_wc.png', sw_list=None, show_wc=True):
    """Compiles and saves WordCloud image as .png file. Option to display in notebook.

    Args:
        wc_list (iterable): Designed with list of strings to be used in WordCloud.
        mask_path (str, optional): File name for WordCloud mask. Defaults to 'Twitter_logo.png'.
        save_path (str, optional): File name for WordCloud image. Defaults to 'img_wc.png'.
        sw_list (iterable, optional): Designed with list of strings to be removed from corpus
            prior to visualizing. Defaults to None.
        show_wc (bool, optional): Enables inline display of WordCLoud. Defaults to True.

    Returns:
        WordCloud Image: Default will save and display WordCloud image. Set 'show_wc' to False 
            for save-only.
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from nltk.tokenize.treebank import TreebankWordDetokenizer
    from keras.preprocessing import text
    from wordcloud import WordCloud, ImageColorGenerator
    from os import path
    from PIL import Image
    
    ### Following code has contributions from https://www.datacamp.com/community/tutorials/wordcloud-python
    
    ## Converting .png into RGB array
    img_mask = np.array(Image.open(mask_path))

    ## Container for transformed array
    img_mask_trnsf = np.ndarray((img_mask.shape[0], img_mask.shape[1]), np.int32)

    ## Mapping function to convert 0s to 255s per row
    for i in range(len(img_mask)):
        img_mask_trnsf[i] = list(map(transform_format_img, img_mask[i]))

    ## Instantiate Keras' tokenizer for word cloud + symbols to filter out
    filt = '!¡÷«»¬©"$£%¢&()*+±,-./:;<=>?@[\\]^_`´{|}~\t\n'
    viz_tkzr = text.Tokenizer(filters=filt)
    
    ## Filtering out stop words
    #sw_reg = '|'.join(sw_list)
    #pattern = r'({})'.format(sw_reg)
    #wc_list = regex_cleaner(wc_list, pattern, '')

    ## Fit, convert to seqeunces + convert back to tokenied text
    viz_tkzr.fit_on_texts(wc_list)
    wc_list_seq = viz_tkzr.texts_to_sequences(wc_list)
    wc_list_tkn = viz_tkzr.sequences_to_texts(wc_list_seq)

    ## Convert tokenized text into one long string
    wc_list_detkn = TreebankWordDetokenizer().detokenize(wc_list_tkn)

    ## Create a word cloud image + generate wordcloud
    wc = WordCloud(background_color="white", max_words=1000, mask=img_mask_trnsf,
                   stopwords=sw_list, contour_width=3, contour_color='midnightblue', colormap='cividis')
    wc.generate(wc_list_detkn)

    ## Save to file + display!
    wc.to_file(save_path)

    if show_wc:
        plt.figure(figsize=[20,10])
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.show();
        
    return 'Done!'
    
def freq_dist_viz(fd_series, topn=20, stopword_extsn=None, show_fd=True, keep_sw=False):
    """Tokenize text data and prepare NLTK FreqDist for corpus. Also will display plot
       of 'topn' most frequent tokens after stopword removal.

    Args:
        fd_series (iterable): Designed with pandas Series filled of strings.
        topn (int, optional): Top number of tokens to display in FreqDist plot. Defaults to 20.
        stopword_extsn (iterable, optional): List of additional stopwords beyond NLTK's
            'stopwords' list to remove from corpus. Defaults to None.
        show_fd (bool, optional): Enables FreqDist plotting. Defaults to True.
        keep_sw (bool, optional): Enables output of 'stop_words' list. Defaults to False.

    Returns:
        FreqDist obj.: Returns nltk.FreqDist() of data and visualize 'topn' tokens. Optional
                           return of stopword list used in tokenizing. 
    """

    import matplotlib.pyplot as plt
    import pandas as pd
    import string
    from nltk import TweetTokenizer, FreqDist
    from nltk.corpus import stopwords
    
    ## Grabbing preset stopword list from nltk 
    stop_words = list(stopwords.words('english'))

    ## Grabbing common punctuation from python + ensuring hashtags survive
    punct = list(string.punctuation)
    punct.remove('#')

    ## Cobiming the two lists + specifics to this dataset
    stop_words.extend(list(string.punctuation))
    stop_words.extend(['#SXSW', '#sxsw', 'RT', 'link', '\x89', 'SXSW', 'sxsw'])
    
    if stopword_extsn:
        stop_words.extend(stopword_extsn)
        print('Stopwords Added:', stopword_extsn)
        print('\n')
        
    ## Instantiating tokenizer + mapping to tweets series
    twt_tkzr = TweetTokenizer(strip_handles=True)
    twt_tkns = fd_series.map(twt_tkzr.tokenize)

    ## Creating new list w/o stopwords + punctuation
    twt_tkns_cln = []
    for twt in twt_tkns:
        low_twt = [word.lower() for word in twt]
        new_twt = [word for word in low_twt if not word in stop_words]
        twt_tkns_cln.append(new_twt)

    ## Container to hold all tokens as-if one string + filling tweet by tweet
    all_tkns = []
    for tweet in twt_tkns_cln:
        all_tkns.extend(tweet)

    ## Using nltk to count frequencies of tokens
    tkn_freq = FreqDist(all_tkns)

    ## Capturing top-n most common tokens
    tkn_freq_topn = tkn_freq.most_common(topn)

    ## Moving to dict for easier conv. to df for visualization
    tkn_freq_topn_dict = {}
    for item in tkn_freq_topn:
        tkn_freq_topn_dict[item[0]] = item[1]

    tkn_freq_topn_df = pd.DataFrame.from_dict(tkn_freq_topn_dict, orient='index')

    ## Visualizing + styling
    if show_fd:
        fig, ax = plt.subplots(figsize=(10,6))
        tkn_freq_topn_df.plot(kind='barh', color='g', title=f'Top {topn} Tokens by Frequency', legend=False, ax=ax)
        ax.grid(alpha=0.3)
        ax.set(xlabel='# of Appearances', ylabel='Token')
        ax.set_facecolor('#e8e8e8')
        plt.show();
        
    if keep_sw:
        return tkn_freq, stop_words
    else:
        return tkn_freq
    
def prep_text_visuals(fd_series, wc_list, topn=20, mask_path='Twitter_logo.png', save_path='img_wc.png', stopword_extsn=None, show_wc=True, show_fd=True, keep_fd=False):
    """Operationalizes '' and '' functions to display in one line of code. Optional return
       of nltk.FreqDist() object from ''.

    Args:
        fd_series (iterable): Designed with pandas Series filled of strings. 
        wc_list (iterable): Designed with list of strings to be used in WordCloud.
        topn (int, optional): Top number of tokens to display in FreqDist plot. Defaults to 20.
        mask_path (str, optional): File name for WordCloud mask. Defaults to 'Twitter_logo.png'.
        save_path (str, optional): File name for WordCloud image. Defaults to 'img_wc.png'.
        stopword_extsn (iterable, optional): List of additional stopwords beyond NLTK's
            'stopwords' list to remove from corpus. Defaults to None.
        show_wc (bool, optional): Enables inline display of WordCLoud. Defaults to True.
        show_fd (bool, optional): Enables FreqDist plotting. Defaults to True.
        keep_fd (bool, optional): Enables output of nltk.FreqDist() object. Defaults to False.

    Returns:
        str: Displays visuals from '' and '' prints string denoting the codes completion. Optional
             return of nltk.FreqDist() object.
    """
    
    #### MAKING TOP-N GRAPH
    
    tkn_freq, stop_words = freq_dist_viz(fd_series, topn=topn, stopword_extsn=stopword_extsn, show_fd=show_fd, keep_sw=True)
    
    #### MAKING WORD CLOUD
    
    word_cloud_viz(wc_list, mask_path=mask_path, save_path=save_path, sw_list=stop_words, show_wc=show_wc)
        
    if keep_fd:
        return tkn_freq
    
    return 'Done!'