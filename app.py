from preprocessing import preprocessing
from cnn import cnn
    
if __name__ == "__main__":
    
    preprocessing_obj = preprocessing()
    train_x,train_y,test_x,test_y = preprocessing_obj.fetch_datasets()
    preprocessing_obj.print_info()
    # preprocessing_obj.plot_sample(999)
    
    cnn_obj = cnn(train_x,train_y,test_x,test_y)
    # cnn_obj.print_model_summary()
    cnn_obj.train_model(120,200)
    cnn_obj.evaluate_model()
    # cnn_obj.save_model("./model_v1")




















































