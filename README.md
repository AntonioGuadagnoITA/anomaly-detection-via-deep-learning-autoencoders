# anomaly-detection-via-deep-learning-autoencoders

## Related Work
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Paper&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Year | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Venue&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AE&nbsp;Usage&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Network&nbsp;Topology&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dataset&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |Regressors| Uni&nbsp;or&nbsp;Multi |Regularity| Granularity |Noise&nbsp;Level |&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Anomaly&nbsp;Type&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Proprietary |    
|----------------------------------|----|-------------------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|---------------------|-------------------------------------------|---------------------------------------------------|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
|Himeur et al. [[1]](#1)           |2021|Appl Energ                           |(Survey paper)                                                   |                                                                                                                                         |                                                                                                                                                                 |       |                     |                                           |                                                   |            |                                                                                                                                                                                  |                                                                                                                                               |
|Malhotra et al. [[2]](#2)         |2016|ICML Anomaly Detection Workshop      |Anomaly Detection                                                |LSTM Encoder-Decoder                                                                                                                     |power demand, space shuttle, and ECG, and two real-world engine datasets with both predictive and unpredictable behavior                                         |no     |univariate           |daily, weekly                              |15 min.                                            |low         |unlabeled - holidays                                                                                                                                                              |public http://www.cs.ucr.edu/~eamonn/discords/                                                                                                 |          
|Fan et al. [[3]](#3)              |2018|Appl Energ                           |Anomaly Detection                                                |Ensamble of fully connected Vanilla AutoEncoders + 1D Convolutional AutoEncoder                                                          |electric consumption data from an educational building in Hong Kong                                                                                              |yes    |univariate           |daily                                      |30 min.                                            |medium-low  |unlabeled - (1) atypical or rare operations (2) transient operations (3) improper control strategies                                                                              |private                                                                                                                                        |          
|Araya et al. [[4]](#4)            |2016|IJCNN                                |Anomaly Detection                                                |Vanilla AutoEncoder                                                                                                                      |HVAC consumption data of a school building                                                                                                                       |yes    |univariate           |daily                                      |5 min.                                             |low         |unlabeled - unspecified real + unspecified synthetic                                                                                                                              |private                                                                                                                                        |          
|Araya et al. [[5]](#5)            |2017|Energ Buildings                      |Anomaly Detection + Dimensionality Reduction                     |Ensamble of methods, including a Vanilla AutoEncoder                                                                                     |HVAC consumption data of a school building                                                                                                                       |yes    |univariate           |daily                                      |5 min.                                             |low         |unlabeled - unspecified real + unspecified synthetic                                                                                                                              |private                                                                                                                                        |          
|Tasfi et al. [[6]](#6)            |2017|iThings, GreenCom, CPSCom, SmartData |Anomaly Detection                                                |Convolutional AutoEncoder for the reconstruction of unlabeled data and a MLP classifier for labeled data                                 |electrical consumption data of buildings                                                                                                                         |no     |univariate           |daily, weekly                              |5 min.                                             |medium-low  |labeled - unspecified real + synthetic anomalies (increase in usage over time for a small duration in periods with typical low usage)                                             |private                                                                                                                                        |          
|Pereira and Silveira [[7]](#7)    |2018|ICMLA                                |Anomaly Detection                                                |Variational Bi-LSTM AutoEncoder with a variational self-attention mechanism                                                              |solar photovoltaic generation time-series                                                                                                                        |no     |univariate           |daily                                      |15 min.                                            |low         |unlabeled - a brief shading, a fault, a spike anomaly, an example of a daily curve where snow covered the surface of the PV panel and a sequence corresponding to a cloudy day    |private                                                                                                                                        |          
|Dai et al. [[8]](#8)              |2022|INTAP                                |Anomaly Detection                                                |Variational LSTM AutoEncoder with a fully connected attention mechanism                                                                  |smart meter dataset about water supply temperature for district heating                                                                                          |yes    |multivariate         |not specified                              |irregular (minute-level)                           |medium-low  |unlabeled - unspecified                                                                                                                                                           |private                                                                                                                                        |          
|Castangia et al. [[9]](#9)        |2022|Sustain Energy, Grids and Netw       |Anomaly Detection                                                |Variational Convolutional AutoEncoder                                                                                                    |household appliances from 6 households in Switzerland                                                                                                            |no     |univariate           |not specified                              |1 sec.                                             |medium      |unlabeled - synthetic anomalies                                                                                                                                                   |public https://www.vs.inf.ethz.ch/res/show.html?what=eco-data                                                                                  |          
|Lee et al. [[10]](#10)            |2022|ICCE                                 |Anomaly Detection                                                |Bi-LSTM Autoncoder                                                                                                                       |data of electricity/water/heating/hot water from 985 households                                                                                                  |no     |univariate           |not specified                              |15 min.                                            |low         |unlabeled - unspecified                                                                                                                                                           |private                                                                                                                                        |          
|Weng et al. [[11]](#11)           |2018|IEEE Access                          |Anomaly Detection                                                |LSTM AutoEncoder                                                                                                                         |electricity consumption data, natural gas and water from a residential house                                                                                     |no     |univariate           |not specified                              |1 min.                                             |low         |labeled - unspecified                                                                                                                                                             |public https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/FIE0S4                                                         |          
|Yuan and Jia [[12]](#12)          |2015|IIH-MSP                              |Feature Extraction                                               |Vanilla AutoEncoder                                                                                                                      |alarm monitoring data, electricty monthly consumption data and other general information about buildings                                                         |yes    |univariate           |not specified                              |1 month                                            |low         |labeled - specified by alarms                                                                                                                                                     |private                                                                                                                                        |          
|Tsai et al. [[13]](#13)           |2022|Electronics                          |Feature Extraction                                               |Vanilla AutoEncoder                                                                                                                      |five residences randomly selected from 200 residential users who had installed energy management systems                                                         |yes    |univariate           |not specified                              |1 min.                                             |low         |unlabeled - extreme power consumption behaviors such as sharp rises or falls                                                                                                      |private                                                                                                                                        |          
|Kaymakci et al. [[14]](#14)       |2021|Procedia CIRP                        |Anomaly Detection                                                |LSTM AutoEncoder                                                                                                                         |(1) electricity consumption of a German metal processing company (2) electricity consumption of a laser-punching machine                                         |no     |univariate           |(1) daily, weekly                          |15 sec.                                            |medium      |unlabeled - (1) high solar power generation (2) significant deviations of the operating mode                                                                                      |private                                                                                                                                        |          
|Ullah et al. [[15]](#15)          |2020|Sensors                              |Feature Extraction                                               |Vanilla AutoEncoder                                                                                                                      |(1) energy consumption data from residential buildings in a city (2) energy consumption data of a single house                                                   |yes    |univariate           |not specified                              |(1) 1 month (2) 1 min.                             |low         |unlabeled - unspecified                                                                                                                                                           |(1) public https://data.openei.org/search?q= (2) public https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption|  
|Chahla et al. [[16]](#16)         |2019|ICPRAM                               |Anomaly Detection                                                |Vanilla AutoEncoder                                                                                                                      |electricity utilization from 67 electronic devices in nearly 1000 houses                                                                                         |no     |univariate           |daily, weekly                              |1 day                                              |low         |unlabeled - unspecified + synthetic                                                                                                                                               |public https://dataport.pecanstreet.org                                                                                                        |
|Kardi et al. [[17]](#17)          |2021|EEEIC/I&CPS Europe                   |Anomaly Detection                                                |LSTM AutoEncoder                                                                                                                         |electricity consumption                                                                                                                                          |yes    |univariate           |daily, weekly, yearly                      |1 hour                                             |low         |unlabeled - unspecified                                                                                                                                                           |private                                                                                                                                        |         
|Chahla et al. [[18]](#18)         |2020|Energ Effic                          |Anomaly Detection                                                |Vanilla AutoEncoder                                                                                                                      |electricity utilization from 67 electronic devices in nearly 1000 houses                                                                                         |no     |univariate           |daily, weekly                              |1 day                                              |low         |unlabeled - unspecified + synthetic                                                                                                                                               |public https://dataport.pecanstreet.org                                                                                                        |          
|Zhao et al. [[19]](#19)           |2021|IEEE Internet Things                 |Anomaly Detection + Denoising                                    |Vanilla AutoEncoder                                                                                                                      |electricity consumption data of six different houses and high-frequency current/voltage data for the main power supply of two of these houses                    |yes    |univariate           |daily, weekly                              |1 sec.                                             |medium-low  |unlabeled + labeled - unspecified                                                                                                                                                 |public http://redd.csail.mit.edu                                                                                                               |          
|Wang et al. [[20]](#20)           |2020|ISGT-Europe                          |Anomaly Detection + Dimensionality Reduction                     |Vanilla Autoencoder                                                                                                                      |electricity consumption data from 361 buildings of a university campus                                                                                           |yes    |univariate           |weekly                                     |15 min.                                            |low         |labeled - faulty smart meters and unusual consumptions                                                                                                                            |private                                                                                                                                        |          
|Wang et al. [[21]](#21)           |2022|IEEE Access                          |Feature Extraction                                               |Vanilla AutoEncoder                                                                                                                      |electricity consumption data of commercial buildings in three different Swedish cities                                                                           |no     |univariate           |yearly                                     |1 hour                                             |low         |labeled - unspecified                                                                                                                                                             |private                                                                                                                                        |          
|Park et al. [[22]](#22)           |2021|Adv Artif Intell Appl Cognit Comput  |Anomaly Detection                                                |Variational Vanilla AutoEncoder                                                                                                          |electricity consumption data of the Lawrence Berkeley National Laboratory building                                                                               |yes    |univariate           |not specified                              |1 min.                                             |low         |unlabeled - synthetic anomalies                                                                                                                                                   |private                                                                                                                                        |      
|Nam et al. [[23]](#23)            |2020|ICTC                                 |Anomaly Detection                                                |LSTM AutoEncoder                                                                                                                         |electricity consumption data of lights, outlets, HVAC systems, elevator                                                                                          |yes    |univariate           |daily, weekly                              |1 hour                                             |medium      |unlabeled - unspecified                                                                                                                                                           |private                                                                                                                                        |  
|Alsalemi et al. [[24]](#24)       |2022|SCS                                  |Anomaly Detection                                                |Vanilla AutoEncoder                                                                                                                      |electricity consumption footprints of different appliances registered in a UK household                                                                          |yes    |univariate           |not specified                              |30 min.                                            |low         |unlabeled - (1) excessive consumption (2) consumption when no inhabitants in the house                                                                                            |public https://jack-kelly.com/data/                                                                                                            |          
|Himeur et al. [[25]](#25)         |2022|BDIoT                                |Anomaly Detection                                                |Vanilla AutoEncoder                                                                                                                      |electricity consumption footprints of different appliances registered in a UK household                                                                          |yes    |univariate           |not specified                              |30 min.                                            |low         |unlabeled - (1) excessive consumption (2) consumption when no inhabitants in the house                                                                                            |public https://jack-kelly.com/data/                                                                                                            |          

<br>

## Network Parameters

### CNN AE

*	window = 672
*	stride = 4
*	latent_dim = 6
    <img align="right"  height="50%" width="50%" src="https://github.com/AntonioGuadagnoITA/anomaly-detection-via-deep-learning-autoencoders/blob/main/img/CONV_AE_REC_E.png"> 
*	epochs = 150
*	batch_size = 8
*	f = 7
*	optimizer = adam
*	First Conv Layer – 16 filters
*	Second Conv Layer – 32 filters
*	Third Conv Layer – 64 filters
*	First ConvTranspose Layer – 64 filters
*	Second ConvTranspose Layer – 32 filters
*	Third ConvTranspose Layer – 16 filters
*	Patience = 15
*	Learning rate = 10<sup>-3</sup>


### CNN VAE
*	window = 672
*	stride = 4
*	M = 200
*	latent_dim = 10
    <img align="right"  height="50%" width="50%" src="https://github.com/AGuadagno/Comparison_Autoencoders_Anomaly_Detection_Electricity_Consumption/blob/main/img/CONV_VAE_REC_E.png"> 
*	epochs = 150
*	batch_size = 8
*	f = 5
*	optimizer = adam
*	First Conv Layer – 16 filters
*	Second Conv Layer – 32 filters
*	Third Conv Layer – 64 filters
*	First ConvTranspose Layer – 64 filters
*	Second ConvTranspose Layer – 32 filters
*	Third ConvTranspose Layer – 16 filters
*	Patience = 15
*	Learning rate = 10<sup>-3</sup>

### LSTM AE
*	window = 672
*	stride = 4
*	latent_dim = 6
    <img align="right"  height="42%" width="42%" src="https://github.com/AGuadagno/Comparison_Autoencoders_Anomaly_Detection_Electricity_Consumption/blob/main/img/LSTM_AE_REC_E.png"> 
*	epochs = 150
*	batch_size = 8
*	optimizer = adam
*	First LSTM Layer (encoder) – 64 memory elements
*	Second LSTM Layer (encoder) – 6 memory elements
*	First LSTM Layer (decoder) – 64 memory elements
*	Patience = 10
*	Learning rate = 10<sup>-3</sup>

### LSTM VAE
*	window = 672
*	stride = 4
*	M = 200
    <img align="right"  height="50%" width="50%" src="https://github.com/AGuadagno/Comparison_Autoencoders_Anomaly_Detection_Electricity_Consumption/blob/main/img/LSTM_VAE_REC_E.png"> 
*	latent_dim = 10
*	epochs = 150
*	batch_size = 8
*	optimizer = adam
*	First LSTM Layer (encoder) – 64 memory elements
*	Second LSTM Layer (encoder) – 10 memory elements
*	First LSTM Layer (decoder) – 64 memory elements
*	Patience = 10
*	Learning rate = 10<sup>-3</sup>
 
### CNN VAE Rec Prob
*	window = 672
*	stride = 4
*	M = 200
*	latent_dim = 10
    <img align="right"  height="50%" width="50%" src="https://github.com/AGuadagno/Comparison_Autoencoders_Anomaly_Detection_Electricity_Consumption/blob/main/img/CONV_VAE_REC_P.png"> 
*	epochs = 150
*	batch_size = 8
*	f = 5
*	optimizer = adam
*	First Conv Layer – 16 filters
*	Second Conv Layer – 32 filters
*	Third Conv Layer – 64 filters
*	First ConvTranspose Layer – 64 filters
*	Second ConvTranspose Layer – 32 filters
*	Third ConvTranspose Layer – 16 filters
*	Patience = 15
*	Learning rate = 10<sup>-3</sup>

### LSTM VAE Rec Prob
*	window = 672
*	stride = 4
*	M = 200
    <img align="right"  height="50%" width="50%"  src="https://github.com/AGuadagno/Comparison_Autoencoders_Anomaly_Detection_Electricity_Consumption/blob/main/img/LSTM_VAE_REC_P.png"> 
*	latent_dim = 10
*	epochs = 150
*	batch_size = 8
*	optimizer = adam
*	First LSTM Layer (encoder) – 64 memory elements
*	Second LSTM Layer (encoder) – 10 memory elements
*	First LSTM Layer (decoder) – 64 memory elements
*	Patience = 10
*	Learning rate = 10<sup>-3</sup>
 
### LSTM VAE Self-Attention Rec Prob
*	window = 672
*	stride = 4
    <img align="right"  height="50%" width="50%" src="https://github.com/AGuadagno/Comparison_Autoencoders_Anomaly_Detection_Electricity_Consumption/blob/main/img/LSTM_ATT_REC_P.png"> 
*	M = 200
*	latent_dim = 10
*	epochs = 150
*	batch_size = 8
*	optimizer = adam
*	First LSTM Layer (encoder) – 64 memory elements
*	First LSTM Layer (decoder) – 64 memory elements
*	Patience = 10
*	Learning rate = 10<sup>-3</sup>

### Bi-LSTM VAE Self-Attention Rec Prob
*	window = 672
    <img align="right"  height="50%" width="50%" src="https://github.com/AGuadagno/Comparison_Autoencoders_Anomaly_Detection_Electricity_Consumption/blob/main/img/Bi-LSTM_VAE_ATT_REC_P.png"> 
*	stride = 4
*	M = 200
*	latent_dim = 10
*	epochs = 150
*	batch_size = 8
*	optimizer = adam
*	First Bi-LSTM Layer (encoder) – 72 memory elements
*	First Bi-LSTM Layer (decoder) – 72 memory elements
*	Patience = 20
*	Learning rate = 10<sup>-3</sup>

### Bi-LSTM VAE Conv Self-Attention Rec Prob
*	window = 672
    <img align="right"  height="50%" width="50%" src="https://github.com/AGuadagno/Comparison_Autoencoders_Anomaly_Detection_Electricity_Consumption/blob/main/img/Bi-LSTM_VAE_CONV_ATT_REC_P.png"> 
*	stride = 4
*	M = 200
*	latent_dim = 10
*	epochs = 150
*	batch_size = 8
*	optimizer = adam
*	First Bi-LSTM Layer (encoder) – 72 memory elements
*	First Bi-LSTM Layer (decoder) – 72 memory elements
*	Patience = 20
*	Learning rate = 10<sup>-3</sup>



## References
<a id="1">[1]</a>  Himeur, Y., Ghanem, K., Alsalemi, A., Bensaali, F., & Amira, A. (2021). Artificial intelligence based anomaly detection of energy consumption in buildings: A review, current trends and new perspectives. Appl Energ, 287, 1-26.

<a id="2">[2]</a>  Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G. (2016). LSTM-based encoder-decoder for multi-sensor anomaly detection. In Proc. ICML Anomaly Detection Workshop (pp. 1-5).

<a id="3">[3]</a>  Fan, C., Xiao, F., Zhao, Y., & Wang, J. (2018). Analytical investigation of autoencoder-based methods for unsupervised anomaly detection in building energy data. Appl Energ, 211, 1123-1135.

<a id="4">[4]</a> Araya, D. B., Grolinger, K., ElYamany, H. F., Capretz, M. A., & Bitsuamlak, G. (2016). Collective contextual anomaly detection framework for smart buildings. In Proc. IJCNN (pp. 511-518).

<a id="5">[5]</a> Araya, D. B., Grolinger, K., ElYamany, H. F., Capretz, M. A., & Bitsuamlak, G. (2017). An ensemble learning framework for anomaly detection in building energy consumption. Energ Buildings, 144, 191-206.

<a id="6">[6]</a> Tasfi, N. L., Higashino, W. A., Grolinger, K., & Capretz, M. A. (2017). Deep neural networks with confidence sampling for electrical anomaly detection. In Proc. iThings, GreenCom, CPSCom, SmartData (pp. 1038-1045).

<a id="7">[7]</a> Pereira, J., & Silveira, M. (2018). Unsupervised anomaly detection in energy time series data using variational recurrent autoencoders with attention. In Proc. ICMLA (pp. 1275-1282).

<a id="8">[8]</a> Dai, W., Liu, X., Heller, A., & Nielsen, P. S. (2022). Smart Meter Data Anomaly Detection Using Variational Recurrent Autoencoders with Attention. In Proc. INTAP (pp. 311-324).

<a id="9">[9]</a> Castangia, M., Sappa, R., Girmay, A. A., Camarda, C., Macii, E., & Patti, E. (2022). Anomaly detection on household appliances based on variational autoencoders. Sustain Energy, Grids and Netw, 32, 1-11.

<a id="10">[10]</a> Lee, S., Jin, H., Nengroo, S. H., Doh, Y., Lee, C., Heo, T., & Har, D. (2022). Smart Metering System Capable of Anomaly Detection by Bi-directional LSTM Autoencoder. In Proc. ICCE (pp. 1-6).

<a id="11">[11]</a> Weng, Y., Zhang, N., & Xia, C. (2018). Multi-agent-based unsupervised detection of energy consumption anomalies on smart campus. IEEE Access, 7, 2169-2178.

<a id="12">[12]</a> Yuan, Y., & Jia, K. (2015). A distributed anomaly detection method of operation energy consumption using smart meter data. In Proc. IIH-MSP (pp. 310-313).

<a id="13">[13]</a> Tsai, C. W., Chiang, K. C., Hsieh, H. Y., Yang, C. W., Lin, J., & Chang, Y. C. (2022). Feature Extraction of Anomaly Electricity Usage Behavior in Residence Using Autoencoder. Electronics, 11(9), 1-24.

<a id="14">[14]</a> Kaymakci, C., Wenninger, S., & Sauer, A. (2021). Energy Anomaly Detection in Industrial Applications with Long Short-term Memory-based Autoencoders. Procedia CIRP, 104, 182-187.

<a id="15">[15]</a> Ullah, A., Haydarov, K., Ul Haq, I., Muhammad, K., Rho, S., Lee, M., & Baik, S. W. (2020). Deep learning assisted buildings energy consumption profiling using smart meter data. Sensors, 20(3), 1-15.

<a id="16">[16]</a> Chahla, C., Snoussi, H., Merghem, L., & Esseghir, M. (2019). A Novel Approach for Anomaly Detection in Power Consumption Data. In Proc. ICPRAM (pp. 483-490).

<a id="17">[17]</a> Kardi, M., AlSkaif, T., Tekinerdogan, B., & Catalao, J. P. (2021). Anomaly Detection in Electricity Consumption Data using Deep Learning. In Proc. EEEIC/I&CPS Europe (pp. 1-6).

<a id="18">[18]</a> Chahla, C., Snoussi, H., Merghem, L., & Esseghir, M. (2020). A deep learning approach for anomaly detection and prediction in power consumption data. Energ Effic, 13(8), 1633-1651.

<a id="19">[19]</a> Zhao, Q., Chang, Z., & Min, G. (2021). Anomaly Detection and Classification of Household Electricity Data: A Time Window and Multilayer Hierarchical Network Approach. IEEE Internet Things, 9(5), 3704-3716.

<a id="20">[20]</a> Wang, L., Turowski, M., Zhang, M., Riedel, T., Beigl, M., Mikut, R., & Hagenmeyer, V. (2020). Point and contextual anomaly detection in building load profiles of a university campus. In Proc. ISGT-Europe (pp. 11-15).

<a id="21">[21]</a> Wang, D., Enlund, T., Trygg, J., Tysklind, M., & Jiang, L. (2022). Toward Delicate Anomaly Detection of Energy Consumption for Buildings: Enhance the Performance From Two Levels. IEEE Access, 10, 31649-31659.

<a id="22">[22]</a> Park, S., Jung, S., Hwang, E., & Rho, S. (2021). Variational AutoEncoder-Based Anomaly Detection Scheme for Load Forecasting. In Proc. Adv Artif Intell Appl Cognit Comput (pp. 833-839).

<a id="23">[23]</a> Nam, H. S., Jeong, Y. K., & Park, J. W. (2020). An anomaly detection scheme based on lstm autoencoder for energy management. In Proc. ICTC (pp. 1445-1447). 

<a id="24">[24]</a> Alsalemi, A., Himeur, Y., Bensaali, F., & Amira, A. (2022). An innovative edge-based Internet of Energy solution for promoting energy saving in buildings. SCS 78, 1-13.

<a id="25">[25]</a> Himeur, Y., Alsalemi, A., Bensaali, F., & Amira, A. (2022). Detection of appliance-level abnormal energy consumption in buildings using autoencoders and micro-moments. In Proc. BDIoT (pp. 179-193).
