import pandas
import numpy
import os
seed = 42
numpy.random.seed(seed)
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

dataframe = pandas.read_csv("1trn")
dataframe_tst = pandas.read_csv("1tst")


float_col = ['frame.time_epoch', 'frame.time_delta', 'frame.time_delta_displayed', 'frame.time_relative']

encoder = LabelEncoder()
encoder.classes_ = numpy.load('classesAWID.npy')
lb = LabelBinarizer()
print(encoder.classes_)
lb.fit([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

to_int16 = ['radiotap.present.reserved', 'wlan.fc.type_subtype', 'wlan.fc.ds', 'wlan_mgt.fixed.capabilities.cfpoll.ap', 'wlan_mgt.fixed.listen_ival', 'wlan_mgt.fixed.status_code', 'wlan_mgt.fixed.timestamp', 
'wlan_mgt.fixed.aid', 'wlan_mgt.fixed.reason_code', 'wlan_mgt.fixed.auth_seq', 'wlan_mgt.fixed.htact', 'wlan_mgt.fixed.chanwidth', 'wlan_mgt.tim.bmapctl.offset', 'wlan_mgt.country_info.environment', 
'wlan_mgt.rsn.capabilities.ptksa_replay_counter', 'wlan_mgt.rsn.capabilities.gtksa_replay_counter', 'wlan.wep.iv', 'wlan.wep.icv', 'wlan.qos.ack' ]
to_drop = ['frame.interface_id', 'frame.dlt', 'wlan.ra', 'wlan.da', 'wlan.ta', 'wlan.sa', 'wlan.bssid', 'wlan.ba.bm', 'wlan_mgt.fixed.current_ap', 
'wlan_mgt.ssid', 'wlan.tkip.extiv', 'wlan.ccmp.extiv', 'radiotap.dbm_antsignal' ]
to_retain = ['frame.time_relative', 'wlan.seq', 'frame.len']


to_drop2 = ['frame.interface_id','frame.dlt','frame.offset_shift','frame.time_delta_displayed','frame.cap_len','frame.marked','frame.ignored','radiotap.version','radiotap.pad','radiotap.length','radiotap.present.tsft','radiotap.present.flags','radiotap.present.rate','radiotap.present.channel','radiotap.present.fhss','radiotap.present.dbm_antsignal','radiotap.present.dbm_antnoise','radiotap.present.lock_quality','radiotap.present.tx_attenuation','radiotap.present.db_tx_attenuation','radiotap.present.dbm_tx_power','radiotap.present.antenna','radiotap.present.db_antsignal','radiotap.present.db_antnoise','radiotap.present.rxflags','radiotap.present.xchannel','radiotap.present.mcs','radiotap.present.ampdu','radiotap.present.vht','radiotap.present.reserved','radiotap.present.rtap_ns','radiotap.present.vendor_ns','radiotap.present.ext','radiotap.flags.cfp','radiotap.flags.preamble','radiotap.flags.wep','radiotap.flags.frag','radiotap.flags.fcs','radiotap.flags.datapad','radiotap.flags.badfcs','radiotap.flags.shortgi','radiotap.channel.freq','radiotap.channel.type.turbo','radiotap.channel.type.cck','radiotap.channel.type.ofdm','radiotap.channel.type.2ghz','radiotap.channel.type.5ghz','radiotap.channel.type.passive','radiotap.channel.type.dynamic','radiotap.channel.type.gfsk','radiotap.channel.type.gsm','radiotap.channel.type.sturbo','radiotap.channel.type.half','radiotap.channel.type.quarter','radiotap.dbm_antsignal','radiotap.antenna','radiotap.rxflags.badplcp','wlan.fc.type_subtype','wlan.fc.version','wlan.fc.ds','wlan.fc.moredata','wlan.fc.order','wlan.ra','wlan.da','wlan.ta','wlan.sa','wlan.bssid','wlan.bar.type','wlan.ba.control.ackpolicy','wlan.ba.control.multitid','wlan.ba.control.cbitmap','wlan.bar.compressed.tidinfo','wlan.ba.bm','wlan.fcs_good','wlan_mgt.fixed.capabilities.ess','wlan_mgt.fixed.capabilities.ibss','wlan_mgt.fixed.capabilities.cfpoll.ap','wlan_mgt.fixed.capabilities.agility','wlan_mgt.fixed.capabilities.apsd','wlan_mgt.fixed.capabilities.radio_measurement','wlan_mgt.fixed.capabilities.dsss_ofdm','wlan_mgt.fixed.capabilities.del_blk_ack','wlan_mgt.fixed.capabilities.imm_blk_ack','wlan_mgt.fixed.listen_ival','wlan_mgt.fixed.current_ap','wlan_mgt.fixed.status_code','wlan_mgt.fixed.timestamp','wlan_mgt.fixed.aid','wlan_mgt.fixed.reason_code','wlan_mgt.fixed.auth_seq','wlan_mgt.fixed.category_code','wlan_mgt.fixed.htact','wlan_mgt.fixed.chanwidth','wlan_mgt.fixed.fragment','wlan_mgt.fixed.sequence','wlan_mgt.tagged.all','wlan_mgt.ssid','wlan_mgt.ds.current_channel','wlan_mgt.tim.dtim_period','wlan_mgt.tim.bmapctl.multicast','wlan_mgt.tim.bmapctl.offset','wlan_mgt.country_info.environment','wlan_mgt.rsn.gcs.type','wlan_mgt.rsn.pcs.count','wlan_mgt.rsn.akms.count','wlan_mgt.rsn.akms.type','wlan_mgt.rsn.capabilities.preauth','wlan_mgt.rsn.capabilities.no_pairwise','wlan_mgt.rsn.capabilities.ptksa_replay_counter','wlan_mgt.rsn.capabilities.gtksa_replay_counter','wlan_mgt.rsn.capabilities.mfpr','wlan_mgt.rsn.capabilities.mfpc','wlan_mgt.rsn.capabilities.peerkey','wlan_mgt.tcprep.trsmt_pow','wlan_mgt.tcprep.link_mrg','wlan.wep.icv','wlan.tkip.extiv','wlan.ccmp.extiv','wlan.qos.tid','wlan.qos.priority','wlan.qos.eosp','wlan.qos.ack','wlan.qos.amsdupresent','wlan.qos.buf_state_indicated','wlan.qos.bit4','wlan.qos.txop_dur_req','wlan.qos.buf_state_indicated']



dataframe.drop(to_drop, axis=1, inplace=True)
dataframe_tst.drop(to_drop, axis=1, inplace=True)


for column in dataframe:
	#print(column)
	#print(dataframe[column].dtype, column)	
	if column in to_int16:
		dataframe[column] = dataframe[column].apply(lambda x: int(str(x), base=16) if x != '?' else x)
		dataframe[column] = dataframe_tst[column].apply(lambda x: int(str(x), base=16) if x != '?' else x)
	if column == 'class':
		dataframe_tst[column] = lb.transform(encoder.transform(dataframe_tst[column]))
		dataframe[column] = lb.transform(encoder.transform(dataframe[column]))

dataframe = dataframe.replace(['?'], [-1])
dataframe_tst = dataframe_tst.replace(['?'], [-1])

dataframe = dataframe.convert_objects(convert_numeric=True)
dataframe_tst = dataframe_tst.convert_objects(convert_numeric=True)
#for column in dataframe:
#	print(dataframe[column].dtype, column)	

dataset = dataframe.values
X = dataset[:,0:-1]
Y = dataset[:,-1]

dataset_tst = dataframe_tst.values
X_tst = dataset[:,0:-1]
Y_tst = dataset[:,-1]


param_test1 = {
 #'max_depth':range(2,10,1),
 #'min_child_weight':range(1,6,1),
 #'subsample':[i/10.0 for i in range(1,11)],
 #'colsample_bytree':[i/10.0 for i in range(1,11)],
 'learning_rate': [i/10.0 for i in range(1,11,2)],
 'n_estimators' : [i for i in range(50, 400, 50)]
}

gsearch1 = GridSearchCV(estimator = XGBClassifier(), scoring='roc_auc', param_grid = param_test1)
grid_result = gsearch1.fit(X, Y)



f = open('xgbresultsR_full_grid_full.txt', 'a')
y_true, y_pred = Y_tst, grid_result.predict(X_tst)
f.write(classification_report(y_true, y_pred))
f.write("Best: %f using %s \n" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    f.write("%f (%f) with: %r \n" % (mean, stdev, param))
