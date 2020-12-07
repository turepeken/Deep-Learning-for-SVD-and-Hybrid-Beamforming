# Hybrid BF based on DNN for rank-k approximation for 4-by-4 mmW system with geometric channel model
#Quantization approach 1 in which combination of step and piecec-wise linear functions are used. 
# Author: Ture Peken

import numpy as np
import scipy.io
from keras.layers import Input, Lambda,Dense, concatenate,Activation, ZeroPadding2D, Flatten, Conv2D, Dropout,MaxPooling2D,BatchNormalization
from keras.models import Model
from keras.optimizers import adam
import tensorflow as tf
from keras.layers import LeakyReLU, ELU
import math as matt
import tensorflow_probability as tfp

def log2(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
  return numerator / denominator

#This function estimates the rate obtained with the estimated beamformers by the DNN based on rank-k approximation
def estimated_rate(y_hat):
    
    Pr_avg=1.4574e-10#Average received power used in generated dataset for geometric channel model, it should be changed when the average received power of generated channel matrices are changed accordingly.
    SNR=1 #Linear SNR value, rate in 0 dB is achieved in this case. In general, SNR=10*log10(SNR in dB)
    No = Pr_avg/SNR #Noise power
    Num_data_stream = Num_Stream 
    
    H = y_hat[:,N_T*N_R*2+2*(N_T+N_R+1)*Num_Stream:] #Real and imaginary components of vectorized real channel matrices for the selected mini-batch.

    y_real_pred =y_hat[:,Num_Stream*2:Num_Stream*2+(N_T+N_R)*Num_Stream]
    y_imag_pred =y_hat[:,Num_Stream*2+(N_T+N_R)*Num_Stream:Num_Stream*2+2*(N_T+N_R)*Num_Stream]

    
    v_pred=[]
    u_pred=[]
            
    H_temp=[]
    
    #Reshape vectorized channels to channel matrices
    for i in range(N_R):
        H_temp.append(tf.complex(H[:,i*N_T:(i+1)*N_T],H[:,(i+1)*N_T:(i+2)*N_T]))
        i=i+2
    
    H_temp=tf.stack(H_temp,axis=2)
    
#Estimated precoder and combiner at the Tx and the Rx are reshaped as matrices from their vectorized versions.
    for i in range(Num_Stream):
        temp_v=tf.complex(y_real_pred[:,i*N_T:(i+1)*N_T], y_imag_pred[:,i*N_T:(i+1)*N_T])
        temp_u=tf.complex(y_real_pred[:,i*N_R+N_T*Num_Stream:(i+1)*N_R+N_T*Num_Stream], y_imag_pred[:,i*N_R+N_T*Num_Stream:(i+1)*N_R+N_T*Num_Stream])        
        v_pred.append(temp_v)
        u_pred.append(temp_u)
    
    v_pred=tf.stack(v_pred,axis=2)   
    u_pred=tf.stack(u_pred,axis=2)

    # Covariance of combiners is calculated
    CovMat_pred = tf.matmul(tf.transpose(u_pred,conjugate=True,perm=[0, 2, 1]), u_pred)
    
    #Rate corresponding to each channel matrice in the mini-batch is calculated, and stored in R_Temp_pred
    Temp_pred = tf.cast(tf.sqrt(Pr_avg/Num_data_stream),tf.complex64)*tf.matmul(tf.matmul(tf.transpose(u_pred,conjugate=True,perm=[0, 2, 1]),H_temp),v_pred)
    Temp_pred2 = (1/No)*tf.matmul(tf.cast(tfp.math.pinv(tf.cast(CovMat_pred,dtype=tf.float32)),dtype=tf.complex64),tf.matmul(Temp_pred,tf.transpose(Temp_pred,conjugate=True,perm=[0, 2, 1])))

    R_Temp_pred = tf.eye(Num_data_stream,dtype=tf.complex64)+Temp_pred2
    
    #Rate is averaged for the selected mini-batch
    R_pred = tf.reduce_mean(tf.abs(log2(tf.matrix_determinant(R_Temp_pred))))

    return R_pred


def loss1(y_true, y_pred):
    
    #Non-negative constants for penalty terms in loss function
    lambda1=0.1
    lambda2=0.1
    
    v_pred=[]
    u_pred=[]
    
    #Real and imaginary components of real and estimated vectorized channel are seperated into different vectors at the beginning.
    y_pred = y_pred[:,:Num_Stream*2+2*(N_T+N_R)*Num_Stream]
    
    y_real_pred =y_pred[:,Num_Stream*2:Num_Stream*2+(N_T+N_R)*Num_Stream]
    y_imag_pred =y_pred[:,Num_Stream*2+(N_T+N_R)*Num_Stream:Num_Stream*2+2*(N_T+N_R)*Num_Stream]
 
    y_true_temp=y_true[:,Num_Stream*2+2*(N_T+N_R)*Num_Stream:Num_Stream*2+2*(N_T+N_R)*Num_Stream+2*(N_T*N_R)]
    
    y_true_real = y_true_temp[:,0:N_R*N_T]
    y_true_imag = y_true_temp[:,N_R*N_T:N_R*N_T*2]
    
    #Real channel matrices are constructed from their vectorized versions.
    chan_app_true = tf.expand_dims(tf.complex(y_true_real,y_true_imag),axis = 1)
    chan_app_true = tf.reshape(chan_app_true,[-1,N_R,N_T])
    
    #Reshape estimated singular values, precoder and combiner by DNN
    sigma_pred=[]
    for i in range(Num_Stream):
        sigma_pred_temp = tf.complex((y_pred[:,2*i]),tf.cast(0.0,tf.float32))
        sigma_pred.append(sigma_pred_temp)
        v_pred_temp = tf.complex(y_real_pred[:,i*N_T:(i+1)*N_T], y_imag_pred[:,i*N_T:(i+1)*N_T])
        u_pred_temp = tf.complex(y_real_pred[:,i*N_R+N_T*Num_Stream:(i+1)*N_R+N_T*Num_Stream], y_imag_pred[:,i*N_R+N_T*Num_Stream:(i+1)*N_R+N_T*Num_Stream])
        v_pred.append(v_pred_temp[:,:])
        u_pred.append(u_pred_temp[:,:])
    
    u_pred=tf.stack(u_pred,axis=2)
    v_pred=tf.stack(v_pred,axis=2)
    sigma_pred=tf.stack(sigma_pred,axis=1)
    
    #Based on estimated singular values, diagonal singular values matrix is constructed.
    sigma_pred_final=[]
    for i in range(batch_size):
        sigma_pred_final.append(tf.linalg.tensor_diag(sigma_pred[i,:]))
        
    sigma_pred_final=tf.stack(sigma_pred_final,axis=0)
        
    #Estimated rank-k aproximation of channel using estimated singular values, precoder and combiner by DNN
    chan_app_pred = tf.matmul(u_pred,tf.matmul(sigma_pred_final,tf.transpose(v_pred,conjugate=True,perm=[0, 2, 1])))
    
    #Norm of real rank-k approximation is calculated for normalization of loss
    err_norm2=tf.sqrt(tf.reduce_sum(tf.square(tf.abs(chan_app_true)),axis=[1,2]))
    
    #Difference between real and estimated rank-k approximations of channel matrices for the selected mini-batch
    diff=tf.sqrt(tf.reduce_sum(tf.square(tf.abs((chan_app_pred-chan_app_true))),axis=[1,2]))
    
    #Second term in loss function is calculated for the selected mini-batch
    orthogonality_constraint_for_u=tf.matmul(u_pred,tf.transpose(u_pred,conjugate=True,perm=[0, 2, 1]))-tf.eye(N_R,dtype=tf.complex64)

    #Third term in loss function is calculated for the selected mini-batch
    orthogonality_constraint_for_v=tf.matmul(v_pred,tf.transpose(v_pred,conjugate=True,perm=[0, 2, 1]))-tf.eye(N_T,dtype=tf.complex64)
   
    #Loss is computed and averaged over channel matrices in the selected mini-batch
    err = tf.reduce_mean((diff/err_norm2)+lambda1*tf.sqrt(tf.reduce_sum(tf.square(tf.abs(orthogonality_constraint_for_u)),axis=[1,2]))+lambda2*tf.sqrt(tf.reduce_sum(tf.square(tf.abs(orthogonality_constraint_for_v)),axis=[1,2])))
        
    return err

# CNN based DNN for rank-k approximation
def model(input_shape,input_shape2,n_y):
    
    #This input stores real channel matrix normalized to values between -1 and 1, this is given to the DNN as the input.
    X_input = Input(input_shape)
    
    #This input stores each rank-1 approximation of channel matrix seperately.
    X_input2 = Input(input_shape2)
    X2 = Flatten()(X_input2)
    
    #This input stores real channel matrix, it is used for estimating rate.
    X_input3 = Input(input_shape)
    X3 = Flatten()(X_input3)
    
    X = ZeroPadding2D((7, 7))(X_input)

    #Currently 2 convolutional layers are active.
    X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv0')(X)
    X = ELU(alpha=1.0)(X)
    
    X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv1')(X)
    X = ELU(alpha=1.0)(X)
    
    # X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv2')(X)
    # X = ELU(alpha=1.0)(X)
    
    # X = Conv2D(128, (3, 3), strides = (1, 1), name = 'conv3')(X)
    # X = ELU(alpha=1.0)(X)
    
    # X = Conv2D(128, (3, 3), strides = (1, 1), name = 'conv4')(X)
    # X = ELU(alpha=1.0)(X)
    
    # X = Conv2D(128, (3, 3), strides = (1, 1), name = 'conv5')(X)
    # X = ELU(alpha=1.0)(X)


    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)
    X = Dropout(0.2)(X)
    
    X = Flatten()(X)
    
    #Unconstrained estimated RF precoder (T_RF), baseband precoder (T_BB), RF combiner (R_RF), baseband combiner (R_BB)
    T_RF = Dense(n_y-2*Num_Stream)(X)
    T_RF = ELU(alpha=1.0)(T_RF)
    T_RF = Dropout(0.1)(T_RF)
    T_RF = Dense((n_y-2*Num_Stream)/4, activation='sigmoid')(T_RF)
    
    T_BB = Dense(n_y-2*Num_Stream)(X)
    T_BB = ELU(alpha=1.0)(T_BB)
    T_BB = Dropout(0.1)(T_BB)
    T_BB = Dense((n_y-2*Num_Stream)/2, activation='linear')(T_BB)
    
    R_RF = Dense(n_y-2*Num_Stream)(X)
    R_RF = ELU(alpha=1.0)(R_RF)
    R_RF = Dropout(0.1)(R_RF)
    R_RF = Dense((n_y-2*Num_Stream)/4, activation='sigmoid')(R_RF)
    
    R_BB = Dense(n_y-2*Num_Stream)(X)
    R_BB = ELU(alpha=1.0)(R_BB)
    R_BB = Dropout(0.1)(R_BB)
    R_BB = Dense((n_y-2*Num_Stream)/2, activation='linear')(R_BB)
    
    X_temp = Dense(2*Num_Stream, activation='linear')(X)
    
    X_in=concatenate([X_temp, T_RF, T_BB, R_RF, R_BB],axis=1)
    
    # myFunc consists of quantization and normalization layers, generate concatenated constrained T_RF, T_BB, R_RF, R_BB.
    X_out=Lambda(myFunc)(X_in)
        
    output_temp_hybrid=concatenate([X_out, X2, X3],axis=1)
   
    model = Model(inputs = [X_input, X_input2,X_input3], outputs = output_temp_hybrid, name='model')
    
    return model

# Use 1-bit phase shifters and piece-wise linear approximations based quantization
def myFunc(x_in):
    
    Nq_bits=1 #number of bit in phase shifter
    alpha=temp_alpha #this value determines the region which is defined by a piece-wise linear function.
    N_S=Num_Stream
  
    Num_directions=pow(2,Nq_bits) #2^(N_q)
    x_temp=x_in[:,0:2*Num_Stream]
    T_RF=x_in[:,2*Num_Stream:2*Num_Stream+N_T*L_T]
    T_BB=x_in[:,2*Num_Stream+N_T*L_T:2*Num_Stream+N_T*L_T+2*N_S*L_T]
    R_RF=x_in[:,2*Num_Stream+N_T*L_T+2*N_S*L_T:2*Num_Stream+N_T*L_T+2*N_S*L_T+N_R*L_R]
    R_BB=x_in[:,2*Num_Stream+N_T*L_T+2*N_S*L_T+N_R*L_R:2*Num_Stream+N_T*L_T+2*N_S*L_T+N_R*L_R+2*N_S*L_R]
    
    sigma_hat = x_temp

    T_BB_real = T_BB[:,0::2]
    T_BB_imag = T_BB[:,1::2]
    
    R_BB_real = R_BB[:,0::2]
    R_BB_imag = R_BB[:,1::2]
    
    
    #Implement piece-wise linear approximations based quantization (quantization approach 1) and generate constrained T_RF and R_RF
    
    #Transform phase values to between 0 and 2^n_Q
    T_RF_angle_rad= 2*tf.constant(matt.pi)*(T_RF)
    kq=T_RF_angle_rad/(2*tf.constant(matt.pi)/Num_directions)
    
    #Phase values in this region are rounded to 0.
    kq_new=tf.where(kq<=1-alpha,tf.floor(kq),kq)
    
    #Depending on alpha value, phase values between (i-alpha) and (i+alpha) are kept same, otherwise rounded to largest integer value smaller than itself. Here, i=1,..,2^N_q-1
    for i in range(Num_directions-2):
        kq_new=tf.where(tf.logical_and(kq>((i+1)+alpha),kq<=((i+2)-alpha)),tf.floor(kq),kq_new)
          
    #If phase values are between 2^N_q-1 and 2^n_Q, they are kept same.
    kq_new=tf.where(tf.logical_and(kq>((Num_directions-1)+alpha),kq<=Num_directions),kq,kq_new)
    
    # Transform phase values to between 0 and 2*pi
    T_RF_angle_rad=kq_new*(2*tf.constant(matt.pi)/Num_directions)
    
    T_RF = tf.sqrt(tf.cast((1.0/N_T),dtype=tf.complex64))*tf.exp(tf.complex(0.0,T_RF_angle_rad)) #Complex elements of T_RF are constructed with calculated phase value and constant modulus=(1/sqrt(N_T))
    
    #Transform phase values to between 0 and 2^n_Q
    R_RF_angle_rad= 2*tf.constant(matt.pi)*(R_RF)
    kq=R_RF_angle_rad/(2*tf.constant(matt.pi)/Num_directions)
    
    #Phase values in this region are rounded to 0.
    kq_new=tf.where(kq<=1-alpha,tf.floor(kq),kq)
    
    #Depending on alpha value, phase values between (i-alpha) and (i+alpha) are kept same, otherwise rounded to largest integer value smaller than itself. Here, i=1,..,2^N_q-1
    for i in range(Num_directions-2):    
        kq_new=tf.where(tf.logical_and(kq>((i+1)+alpha),kq<=((i+2)-alpha)),tf.floor(kq),kq_new)
    
    #If phase values are between 2^N_q-1 and 2^n_Q, they are kept same.
    kq_new=tf.where(tf.logical_and(kq>((Num_directions-1)+alpha),kq<=Num_directions),kq,kq_new)
    
    # Transform phase values to between 0 and 2*pi
    R_RF_angle_rad=kq_new*(2*tf.constant(matt.pi)/Num_directions)  
    R_RF = tf.sqrt(tf.cast((1.0/N_R),dtype=tf.complex64))*tf.exp(tf.complex(0.0,R_RF_angle_rad))#Complex elements of R_RF are constructed with calculated phase value and constant modulus=(1/sqrt(N_R))
    
        #T_RF is constructed as matrix
    T_RF_tmp=[]
    for i in range(L_T):
        T_RF_tmp.append(T_RF[:,i*N_T:(i+1)*N_T])
    
    T_RF_tmp=tf.stack(T_RF_tmp,axis=2)
    
    R_RF_tmp=[]
    for i in range(L_R):
        R_RF_tmp.append(R_RF[:,i*N_R:(i+1)*N_R])
  
      #R_RF is constructed as matrix
    R_RF_tmp=tf.stack(R_RF_tmp,axis=2)
    
    T_BB = tf.complex(T_BB_real[:,:], T_BB_imag[:,:])  
    R_BB = tf.complex(R_BB_real[:,:], R_BB_imag[:,:])
    
    T_BB_tmp=[]
    R_BB_tmp=[]
    
    for i in range(N_S):
        T_BB_tmp.append(T_BB[:,i*L_T:(i+1)*L_T])
        R_BB_tmp.append(R_BB[:,i*L_R:(i+1)*L_R])

        
    T_BB_tmp=tf.stack(T_BB_tmp,axis=2)
    R_BB_tmp=tf.stack(R_BB_tmp,axis=2)
    
    #Implement normalization layers and generate constrained T_BB and R_BB
    T_BB_tmp=tf.sqrt(tf.cast(N_S,dtype=tf.complex64))*T_BB_tmp/tf.expand_dims(tf.expand_dims(tf.norm(tf.matmul(T_RF_tmp,T_BB_tmp),axis=[1,2]),axis=1),axis=2)
    R_BB_tmp=tf.sqrt(tf.cast(N_S,dtype=tf.complex64))*R_BB_tmp/tf.expand_dims(tf.expand_dims(tf.norm(tf.matmul(R_RF_tmp,R_BB_tmp),axis=[1,2]),axis=1),axis=2)
            
    F=tf.matmul(T_RF_tmp,T_BB_tmp)
    R=tf.matmul(R_RF_tmp,R_BB_tmp)
    
    F_final=F[:,:,0]
    R_final=R[:,:,0]
    
    #T_final=T_RF*T_BB and R_final=R_RF*R_BB are constructed, they are used for approximation of right and left singular vectors matrices.
    for i in range(N_S-1):
        F_final=tf.concat([F_final,F[:,:,i+1]],axis=1)
        R_final=tf.concat([R_final,R[:,:,i+1]],axis=1)
        
    out= concatenate([sigma_hat,tf.real(F_final), tf.real(R_final),tf.imag(F_final), tf.imag(R_final)],axis=1)
            
    return out

L_T=4 #Number of RF chains in transmitter RF precoder
L_R=4 #Number of RF chains in receiver RF combiner

N_T=4 #Number of transmitter antennas
N_R=4 #Number of receiver antennas

Num_Stream=4 #Number of data streams
 
#Obtains train and test 4-by-4 channel matrices generated using geometric channel model
D_train = np.load('./D_44_train.npy')
Y_train = np.load('./Y_44_train.npy')
Z_train = np.load('./Z_44_train.npy')
X_train = np.load('./X_44_train.npy')
C_train = np.load('./C_44_train.npy')

D_test = np.load('./D_44_test.npy')
Y_test = np.load('./Y_44_test.npy')
Z_test = np.load('./Z_44_test.npy')
X_test = np.load('./X_44_test.npy')
C_test = np.load('./C_44_test.npy')

temp_X=X_train
temp_Y=Y_train
temp_Z=Z_train

temp_X_Test=X_test
temp_Y_Test=Y_test
temp_Z_Test=Z_test

temp_H=C_train
temp_H_out=D_train

temp_HTest=C_test
temp_HTest_out=D_test

m,n_H,n_W=temp_X.shape
m_test,_,_=temp_X_Test.shape

#Prepare input and output for training data.
shape_X=(m,n_H,n_W,1)
X_train2=np.zeros(shape_X) #This input stores real channel matrix normalized to values between -1 and 1, this is given to the DNN as the input.
Z_train2=np.zeros((m,n_H*n_W,1)) #This input stores each rank-1 approximation of channel matrix seperately.
Z_train2[:,:,0]=np.squeeze(temp_Z)
Z_train2_out=np.squeeze(Z_train2[:,:,0])

X_train2[:,:,:,0]=temp_X
Y_train2=np.squeeze(temp_Y) # Output for the DNN, this has singular values and vectors for given channel matrix.

H_train2=np.zeros(shape_X)
H_train2[:,:,:,0]=temp_H #This input stores real channel matrix, it is used for estimating rate.
H_train2_out=np.squeeze(temp_H_out) #This output stores real channel matrix, it is used for estimating rate.

#Prepare input and output for test data.
shape_XTest=(m_test,n_H,n_W,1)
X_test2=np.zeros(shape_XTest) #This input stores real channel matrix normalized to values between -1 and 1, this is given to the DNN as the input.
Z_test2=np.zeros((m_test,n_H*n_W,1)) 
Z_test2[:,:,0]=np.squeeze(temp_Z_Test) #This input stores each rank-1 approximation of channel matrix seperately.
Z_test2_out=np.squeeze(Z_test2[:,:,0])

X_test2[:,:,:,0]=temp_X_Test
Y_test2=np.squeeze(temp_Y_Test) # Output for the DNN, this has singular values and vectors for given channel matrix.

H_test2=np.zeros(shape_XTest)
H_test2[:,:,:,0]=temp_HTest #This input stores real channel matrix, it is used for estimating rate.
H_test2_out=np.squeeze(temp_HTest_out) #This output stores real channel matrix, it is used for estimating rate. 

input_shape3 = (n_H,n_W,1)
input_shape33 = (n_H*n_W,1)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

alpha=tf.placeholder(tf.float32)
temp_alpha=tf.zeros_like(alpha)
temp_alpha=sess.run(temp_alpha,feed_dict={alpha:0.05}) 

model_out = model(input_shape3,input_shape33,2*(N_T+N_R+1)*Num_Stream)

batch_size=32

#Adam optimizer with learning rate=0.000001.
model_out.compile(optimizer = adam(lr=0.000001), loss = loss1)

#Total number of channel matrices are 50000
idx = np.arange(0,40000)
idx_test = np.arange(0,10000)

final_loss_log_train=[]
final_loss_log_test=[]
final_rate_log_test=[]
final_rate_log_train=[]

#DNN is trained 1000 iterations. Training and test error are obtained after each iteration. Each time a mini-batch with a size of 32 is selected randomly from training and test datasets.
for i in range(1000):
    np.random.shuffle(idx)
    np.random.shuffle(idx_test)
    idx = idx[:batch_size] 
    idx_test = idx_test[:batch_size] 
    data_shuffle_X_train2 = X_train2[idx,:,:,:]
    data_shuffle_H_train2 = H_train2[idx,:,:,:]
    data_shuffle_Z_train2 = Z_train2[idx,:,:]    
    data_shuffle_Y_train2 = Y_train2[idx,:]
    data_shuffle_Z_train2_out = Z_train2_out[idx,:]
    data_shuffle_H_train2_out = H_train2_out[idx,:]
    data_shuffle_X_test2 = X_test2[idx_test,:,:,:]
    data_shuffle_H_test2 = H_test2[idx_test,:,:,:]
    data_shuffle_Z_test2 = Z_test2[idx_test,:,:]
   
    data_shuffle_Y_test2 = Y_test2[idx_test,:]
    data_shuffle_Z_test2_out = Z_test2_out[idx_test,:]
    data_shuffle_H_test2_out = H_test2_out[idx_test,:]
    data_shuffle_x=[data_shuffle_X_train2,data_shuffle_Z_train2,data_shuffle_H_train2]
    data_shuffle_y=np.concatenate((data_shuffle_Y_train2,data_shuffle_Z_train2_out,data_shuffle_H_train2_out),axis=1)
    data_shuffle_x_test=[data_shuffle_X_test2,data_shuffle_Z_test2,data_shuffle_H_test2]
    data_shuffle_y_test=np.concatenate((data_shuffle_Y_test2,data_shuffle_Z_test2_out,data_shuffle_H_test2_out),axis=1)

    temp_alpha = tf.zeros_like(alpha)
    temp_alpha=sess.run(temp_alpha, feed_dict={alpha: 0.05})
    train_error=model_out.train_on_batch(data_shuffle_x, data_shuffle_y)

    temp_alpha = tf.zeros_like(alpha)
    temp_alpha=sess.run(temp_alpha, feed_dict={alpha: 0.0})
    test_error=model_out.test_on_batch(data_shuffle_x_test, data_shuffle_y_test)

    final_loss_log_train=final_loss_log_train+[[train_error]]
    final_loss_log_test=final_loss_log_test+[[test_error]]

    if i%20 == 0:
        y_hat_test = model_out.predict(data_shuffle_x_test) 
        temp_alpha = tf.zeros_like(alpha)
        temp_alpha=sess.run(temp_alpha, feed_dict={alpha: 0.05})
        y_hat_train = model_out.predict(data_shuffle_x) 

        est_rate = estimated_rate(y_hat_test)
        est_rate_train = estimated_rate(y_hat_train)

        with sess.as_default():
            est_rate=est_rate.eval()
            est_rate_train=est_rate_train.eval()

        final_rate_log_test=final_rate_log_test+[[est_rate]]
        final_rate_log_train=final_rate_log_train+[[est_rate_train]]

        print(est_rate)


        
np.savetxt('rate_with_test_data.csv',final_rate_log_test,delimiter=",",fmt="%s")   
np.savetxt('rate_with_train_data.csv',final_rate_log_train,delimiter=",",fmt="%s")    
  
np.savetxt('loss_with_train_data.csv',final_loss_log_train,delimiter=",",fmt="%s")    
np.savetxt('loss_with_test_data.csv',final_loss_log_test,delimiter=",",fmt="%s")    
 
model_out.save_weights("model.h5")
