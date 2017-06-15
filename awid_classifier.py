import pandas as pd
import numpy as np
import os
from sklearn.externals import joblib
import pyshark
import xgboost
from keras.models import load_model
import matplotlib.pyplot as plt

class AWIDClassifier(object):

	def __init__(self, model, isKeras, reducedFeatures):
		self.model = model
		self.isKerasModel = isKeras
		self.reducedFeatures = reducedFeatures
		self.to_int16 = ['radiotap.present.reserved', 'wlan.fc.type_subtype', 'wlan.fc.ds', 'wlan_mgt.fixed.capabilities.cfpoll.ap', 'wlan_mgta.fixed.listen_ival', 'wlan_mgt.fixed.status_code', 'wlan_mgt.fixed.timestamp', 
		'wlan_mgt.fixed.aid', 'wlan_mgt.fixed.reason_code', 'wlan_mgt.fixed.auth_seq', 'wlan_mgt.fixed.htact', 'wlan_mgt.fixed.chanwidth', 'wlan_mgt.tim.bmapctl.offset', 'wlan_mgt.country_info.environment', 
		'wlan_mgt.rsn.capabilities.ptksa_replay_counter', 'wlan_mgt.rsn.capabilities.gtksa_replay_counter', 'wlan.wep.iv', 'wlan.wep.icv', 'wlan.qos.ack' ]

		self.to_drop = ['frame.interface_id', 'wlan.ra', 'wlan.da', 'wlan.ta', 'wlan.sa', 'wlan.bssid', 'wlan.ba.bm', 'wlan_mgt.fixed.current_ap', 
		'wlan_mgt.ssid', 'wlan.tkip.extiv', 'wlan.ccmp.extiv', 'radiotap.dbm_antsignal' ]
		self.to_drop2 = ['frame.interface_id', 'frame.offset_shift','frame.time_delta_displayed','frame.cap_len','frame.marked','frame.ignored','radiotap.version','radiotap.pad','radiotap.length',
		'radiotap.present.tsft','radiotap.present.flags','radiotap.present.rate','radiotap.present.channel','radiotap.present.fhss','radiotap.present.dbm_antsignal','radiotap.present.dbm_antnoise',
		'radiotap.present.lock_quality','radiotap.present.tx_attenuation','radiotap.present.db_tx_attenuation','radiotap.present.dbm_tx_power','radiotap.present.antenna','radiotap.present.db_antsignal',
		'radiotap.present.db_antnoise','radiotap.present.rxflags','radiotap.present.xchannel','radiotap.present.mcs','radiotap.present.ampdu','radiotap.present.vht','radiotap.present.reserved',
		'radiotap.present.rtap_ns','radiotap.present.vendor_ns','radiotap.present.ext','radiotap.flags.cfp','radiotap.flags.preamble','radiotap.flags.wep','radiotap.flags.frag','radiotap.flags.fcs',
		'radiotap.flags.datapad','radiotap.flags.badfcs','radiotap.flags.shortgi','radiotap.channel.freq','radiotap.channel.type.turbo','radiotap.channel.type.cck','radiotap.channel.type.ofdm',
		'radiotap.channel.type.2ghz','radiotap.channel.type.5ghz','radiotap.channel.type.passive','radiotap.channel.type.dynamic','radiotap.channel.type.gfsk','radiotap.channel.type.gsm',
		'radiotap.channel.type.sturbo','radiotap.channel.type.half','radiotap.channel.type.quarter','radiotap.dbm_antsignal','radiotap.antenna','radiotap.rxflags.badplcp','wlan.fc.type_subtype',
		'wlan.fc.version','wlan.fc.ds','wlan.fc.moredata','wlan.fc.order','wlan.ra','wlan.da','wlan.ta','wlan.sa','wlan.bssid','wlan.bar.type','wlan.ba.control.ackpolicy','wlan.ba.control.multitid',
		'wlan.ba.control.cbitmap','wlan.bar.compressed.tidinfo','wlan.ba.bm','wlan.fcs_good','wlan_mgt.fixed.capabilities.ess','wlan_mgt.fixed.capabilities.ibss','wlan_mgt.fixed.capabilities.cfpoll.ap',
		'wlan_mgt.fixed.capabilities.agility','wlan_mgt.fixed.capabilities.apsd','wlan_mgt.fixed.capabilities.radio_measurement','wlan_mgt.fixed.capabilities.dsss_ofdm','wlan_mgt.fixed.capabilities.del_blk_ack',
		'wlan_mgt.fixed.capabilities.imm_blk_ack','wlan_mgt.fixed.listen_ival','wlan_mgt.fixed.current_ap','wlan_mgt.fixed.status_code','wlan_mgt.fixed.timestamp','wlan_mgt.fixed.aid','wlan_mgt.fixed.reason_code',
		'wlan_mgt.fixed.auth_seq','wlan_mgt.fixed.category_code','wlan_mgt.fixed.htact','wlan_mgt.fixed.chanwidth','wlan_mgt.fixed.fragment','wlan_mgt.tagged.all','wlan_mgt.ssid','wlan_mgt.ds.current_channel',
		'wlan_mgt.tim.dtim_period','wlan_mgt.tim.bmapctl.multicast','wlan_mgt.tim.bmapctl.offset','wlan_mgt.country_info.environment','wlan_mgt.rsn.gcs.type','wlan_mgt.rsn.pcs.count','wlan_mgt.rsn.akms.count',
		'wlan_mgt.rsn.akms.type','wlan_mgt.rsn.capabilities.preauth','wlan_mgt.rsn.capabilities.no_pairwise','wlan_mgt.rsn.capabilities.ptksa_replay_counter','wlan_mgt.rsn.capabilities.gtksa_replay_counter',
		'wlan_mgt.rsn.capabilities.mfpr','wlan_mgt.rsn.capabilities.mfpc','wlan_mgt.rsn.capabilities.peerkey','wlan_mgt.tcprep.trsmt_pow','wlan_mgt.tcprep.link_mrg','wlan.wep.icv','wlan.tkip.extiv','wlan.ccmp.extiv',
		'wlan.qos.tid','wlan.qos.priority','wlan.qos.eosp','wlan.qos.ack','wlan.qos.amsdupresent','wlan.qos.buf_state_indicated','wlan.qos.bit4','wlan.qos.txop_dur_req','wlan.qos.buf_state_indicated']
		self.to_drop2_pcap = ['frame.interface_id', 'frame.offset_shift','frame.time_delta_displayed','frame.cap_len','frame.marked','frame.ignored','radiotap.version','radiotap.pad','radiotap.length',
		'radiotap.present.tsft','radiotap.present.flags','radiotap.present.rate','radiotap.present.channel','radiotap.present.fhss','radiotap.present.dbm_antsignal','radiotap.present.dbm_antnoise',
		'radiotap.present.lock_quality','radiotap.present.tx_attenuation','radiotap.present.db_tx_attenuation','radiotap.present.dbm_tx_power','radiotap.present.antenna','radiotap.present.db_antsignal',
		'radiotap.present.db_antnoise','radiotap.present.rxflags','radiotap.present.xchannel','radiotap.present.mcs','radiotap.present.ampdu','radiotap.present.vht','radiotap.present.reserved',
		'radiotap.present.rtap_ns','radiotap.present.vendor_ns','radiotap.present.ext','radiotap.flags.cfp','radiotap.flags.preamble','radiotap.flags.wep','radiotap.flags.frag','radiotap.flags.fcs',
		'radiotap.flags.datapad','radiotap.flags.badfcs','radiotap.flags.shortgi','radiotap.channel.freq','radiotap.channel.flags.turbo','radiotap.channel.flags.cck','radiotap.channel.flags.ofdm',
		'radiotap.channel.flags.2ghz','radiotap.channel.flags.5ghz','radiotap.channel.flags.passive','radiotap.channel.flags.dynamic','radiotap.channel.flags.gfsk','radiotap.channel.flags.gsm',
		'radiotap.channel.flags.sturbo','radiotap.channel.flags.half','radiotap.channel.flags.quarter','radiotap.dbm_antsignal','radiotap.antenna','radiotap.rxflags.badplcp','wlan.fc.type_subtype',
		'wlan.fc.version','wlan.fc.ds','wlan.fc.moredata','wlan.fc.order','wlan.ra','wlan.da','wlan.ta','wlan.sa','wlan.bssid','wlan.bar.type','wlan.ba.control.ackpolicy','wlan.ba.control.multitid',
		'wlan.ba.control.cbitmap','wlan.bar.compressed.tidinfo','wlan.ba.bm','wlan.fcs_good','wlan_mgt.fixed.capabilities.ess','wlan_mgt.fixed.capabilities.ibss','wlan_mgt.fixed.capabilities.cfpoll.ap',
		'wlan_mgt.fixed.capabilities.agility','wlan_mgt.fixed.capabilities.apsd','wlan_mgt.fixed.capabilities.radio_measurement','wlan_mgt.fixed.capabilities.dsss_ofdm','wlan_mgt.fixed.capabilities.del_blk_ack',
		'wlan_mgt.fixed.capabilities.imm_blk_ack','wlan_mgt.fixed.listen_ival','wlan_mgt.fixed.current_ap','wlan_mgt.fixed.status_code','wlan_mgt.fixed.timestamp','wlan_mgt.fixed.aid','wlan_mgt.fixed.reason_code',
		'wlan_mgt.fixed.auth_seq','wlan_mgt.fixed.category_code','wlan_mgt.fixed.htact','wlan_mgt.fixed.chanwidth','wlan_mgt.fixed.fragment','wlan_mgt.tagged.all','wlan_mgt.ssid','wlan_mgt.ds.current_channel',
		'wlan_mgt.tim.dtim_period','wlan_mgt.tim.bmapctl.multicast','wlan_mgt.tim.bmapctl.offset','wlan_mgt.country_info.environment','wlan_mgt.rsn.gcs.type','wlan_mgt.rsn.pcs.count','wlan_mgt.rsn.akms.count',
		'wlan_mgt.rsn.akms.type','wlan_mgt.rsn.capabilities.preauth','wlan_mgt.rsn.capabilities.no_pairwise','wlan_mgt.rsn.capabilities.ptksa_replay_counter','wlan_mgt.rsn.capabilities.gtksa_replay_counter',
		'wlan_mgt.rsn.capabilities.mfpr','wlan_mgt.rsn.capabilities.mfpc','wlan_mgt.rsn.capabilities.peerkey','wlan_mgt.tcprep.trsmt_pow','wlan_mgt.tcprep.link_mrg','wlan.wep.icv','wlan.tkip.extiv','wlan.ccmp.extiv',
		'wlan.qos.tid','wlan.qos.priority','wlan.qos.eosp','wlan.qos.ack','wlan.qos.amsdupresent','wlan.qos.buf_state_indicated','wlan.qos.bit4','wlan.qos.txop_dur_req','wlan.qos.buf_state_indicated.1']
		self.float_col = ['frame.time_epoch', 'frame.time_delta', 'frame.time_delta_displayed', 'frame.time_relative']
		self.bools = [
			"wlan.fc.frag", "wlan.fc.retry", "wlan.fc.pwrmgt", "wlan.fc.protected", "wlan_mgt.fixed.capabilities.privacy", 
			"wlan_mgt.fixed.capabilities.preamble", "wlan_mgt.fixed.capabilities.pbcc", "wlan_mgt.fixed.capabilities.spec_man", 
			"wlan_mgt.fixed.capabilities.short_slot_time", "wlan.qos.buf_state_indicated", 'frame.marked', 'radiotap.present.tsft',
			"radiotap.present.flags", "radiotap.present.rate", "radiotap.present.channel", "radiotap.present.fhss", "radiotap.present.dbm_antsignal",
			"radiotap.present.dbm_antnoise", "radiotap.present.lock_quality", "radiotap.present.tx_attenuation",
			"radiotap.present.dbm_tx_power", "radiotap.present.db_antsignal", "radiotap.present.db_antnoise", "radiotap.present.rxflags",
			"radiotap.present.xchannel", "radiotap.present.mcs", "radiotap.present.ampdu", "radiotap.present.vht",
			"radiotap.present.rtap_ns", "radiotap.present.ext", "radiotap.flags.cfp", "radiotap.flags.preamble", 
			"radiotap.flags.wep","radiotap.flags.frag","radiotap.flags.fcs","radiotap.flags.datapad","radiotap.flags.badfcs",
			"radiotap.flags.shortgi", "radiotap.channel.flags.turbo","radiotap.channel.flags.cck",
			"radiotap.channel.flags.ofdm",
			"radiotap.channel.flags.2ghz","radiotap.channel.flags.5ghz","radiotap.channel.flags.passive",
			"radiotap.channel.flags.dynamic","radiotap.channel.flags.gfsk","radiotap.channel.flags.gsm",
			"radiotap.channel.flags.sturbo","radiotap.channel.flags.half","radiotap.channel.flags.quarter", 
			"radiotap.rxflags.badplcp", "wlan.fc.moredata", "wlan.fc.order", "wlan.ba.control.ackpolicy", "wlan.ba.control.multitid",
			"wlan.ba.control.cbitmap", "wlan.fcs_good", "wlan_mgt.fixed.capabilities.ess", "wlan_mgt.fixed.capabilities.ibss",
			"wlan_mgt.fixed.capabilities.cfpoll.ap", "wlan_mgt.fixed.capabilities.agility", "wlan_mgt.fixed.capabilities.apsd",
			"wlan_mgt.fixed.capabilities.radio_measurement", "wlan_mgt.fixed.capabilities.dsss_ofdm", "wlan_mgt.fixed.capabilities.del_blk_ack",
			"wlan_mgt.fixed.capabilities.imm_blk_ack", "wlan_mgt.tim.bmapctl.multicast", "wlan_mgt.rsn.capabilities.preauth",
			"wlan_mgt.rsn.capabilities.no_pairwise", "wlan_mgt.rsn.capabilities.mfpr", "wlan_mgt.rsn.capabilities.mfpc",
			"wlan_mgt.rsn.capabilities.peerkey", "wlan.qos.eosp", "wlan.qos.amsdupresent", "wlan.qos.buf_state_indicated",
			"wlan.qos.bit4", "wlan.qos.buf_state_indicated.1"
		]
		self.classes = ['normal', 'arp', 'cafe_latte', 'amok', 'deauthentication', 'authentication_request', 'evil_twin', 'beacon', 'probe_response', 'fragmentation'
		,'probe_request', 'chop_chop', 'rts', 'cts', 'hirte', 'power_saving', 'disassociation']
		self.df_in_use = None

	def encode(self, a, t=0):
		e = {
			"normal": ([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 0),
			"arp":    ([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 1),
			"cafe_latte": ([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 2),
			"amok":   ([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], 3),
			"deauthentication": ([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0], 4),
			"authentication_request": ([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], 5),
			"evil_twin": ([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], 6),
			"beacon": ([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0], 7),
			"probe_response": ([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], 8),
			"fragmentation": ([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0], 9),
			"probe_request": ([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0], 10),
			"chop_chop":([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0], 11),
			"rts":([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], 12),
			"cts":([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], 13),
			"hirte":([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0], 14),
			"power_saving":([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0], 15),
			"disassociation":([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], 16)
		}
		return(e[a][t])


	def preprocessing(self, filename, pcap):
		with open(filename, 'r') as f:
			with open('preprocessed.csv', 'w') as ff:
				for line in f:
					if len(line.split(',')) != 152:
						pass
					else:
						ff.write(line + '\n')


		self.df_in_use = pd.read_csv('preprocessed.csv')
		self.df_in_use.rename(columns={self.df_in_use.columns[150]: 'wlan.qos.buf_state_indicated.1'}, inplace=True)
		
		if self.reducedFeatures == False:
			self.df_in_use.drop(self.to_drop, axis=1, inplace=True)
		else:
			if pcap == True:
				self.df_in_use.drop(self.to_drop2_pcap, axis=1, inplace=True)
			else:
				self.df_in_use.drop(self.to_drop2, axis=1, inplace=True)
		
		self.df_in_use.replace("", -1, inplace=True)
		self.df_in_use.replace("?", -1, inplace=True)
		self.df_in_use.replace(np.nan, -1, inplace=True)
		for a in self.df_in_use.columns:
			if a != "class":
				try:
					self.df_in_use[a] = self.df_in_use[a].apply(float)
				except:
					self.df_in_use[a] = self.df_in_use[a].apply(lambda x: float(int(str(x), base=16)))
				finally:
					pass

		for a in self.bools:
			if a in self.df_in_use.columns:
				self.df_in_use[a+".is0"] = self.df_in_use[a].apply(lambda x: 1 if x == 0 else 0)
				self.df_in_use[a+".is1"] = self.df_in_use[a].apply(lambda x: 1 if x == 1 else 0)
				self.df_in_use[a+".is-1"] = self.df_in_use[a].apply(lambda x: 1 if x == -1 else 0)

		self.df_in_use.drop(self.bools, axis=1, inplace=True, errors='ignore')

		return self.df_in_use.values


	def graph(self, chart_info):


		ind = np.arange(17) 
		width = 1
		fig, ax = plt.subplots()
		fig.set_size_inches(18, 22)
		r = ax.bar(ind, chart_info, width, color='b')

		for i in range(len(r)):
			height = r[i].get_height()
			ax.text(r[i].get_x() + r[i].get_width()/2., 1.03*height,
        	        '%d' % int(height),
        	        ha='center', va='bottom')


		ax.set_ylabel('Number of packets')
		ax.set_title('Detected classes of packets in file')
		ax.set_xticks(ind + width / 2)
		ax.set_xticklabels(self.classes)
		plt.show()



	def get_attacks_info_from_pcap(self, filename):
		command = 'tshark -r ' + filename +' -T fields -e frame.interface_id -e frame.offset_shift -e frame.time_epoch -e frame.time_delta -e frame.time_delta_displayed -e frame.time_relative -e frame.len -e frame.cap_len -e frame.marked -e frame.ignored -e radiotap.version -e radiotap.pad -e radiotap.length -e radiotap.present.tsft -e radiotap.present.flags -e radiotap.present.rate -e radiotap.present.channel -e radiotap.present.fhss -e radiotap.present.dbm_antsignal -e radiotap.present.dbm_antnoise -e radiotap.present.lock_quality -e radiotap.present.tx_attenuation -e radiotap.present.db_tx_attenuation -e radiotap.present.dbm_tx_power -e radiotap.present.antenna -e radiotap.present.db_antsignal -e radiotap.present.db_antnoise -e radiotap.present.rxflags -e radiotap.present.xchannel -e radiotap.present.mcs -e radiotap.present.ampdu -e radiotap.present.vht -e radiotap.present.reserved -e radiotap.present.rtap_ns -e radiotap.present.vendor_ns -e radiotap.present.ext -e radiotap.mactime -e radiotap.flags.cfp -e radiotap.flags.preamble -e radiotap.flags.wep -e radiotap.flags.frag -e radiotap.flags.fcs -e radiotap.flags.datapad -e radiotap.flags.badfcs -e radiotap.flags.shortgi -e radiotap.datarate -e radiotap.channel.freq -e radiotap.channel.flags.turbo -e radiotap.channel.flags.cck -e radiotap.channel.flags.ofdm -e radiotap.channel.flags.2ghz -e radiotap.channel.flags.5ghz -e radiotap.channel.flags.passive -e radiotap.channel.flags.dynamic -e radiotap.channel.flags.gfsk -e radiotap.channel.flags.gsm -e radiotap.channel.flags.sturbo -e radiotap.channel.flags.half -e radiotap.channel.flags.quarter -e radiotap.dbm_antsignal -e radiotap.antenna -e radiotap.rxflags.badplcp -e wlan.fc.type_subtype -e wlan.fc.version -e wlan.fc.type -e wlan.fc.subtype -e wlan.fc.ds -e wlan.fc.frag -e wlan.fc.retry -e wlan.fc.pwrmgt -e wlan.fc.moredata -e wlan.fc.protected -e wlan.fc.order -e wlan.duration -e wlan.ra -e wlan.da -e wlan.ta -e wlan.sa -e wlan.bssid -e wlan.frag -e wlan.seq -e wlan.bar.type -e wlan.ba.control.ackpolicy -e wlan.ba.control.multitid -e wlan.ba.control.cbitmap -e wlan.bar.compressed.tidinfo -e wlan.ba.bm -e wlan.fcs_good -e wlan_mgt.fixed.capabilities.ess -e wlan_mgt.fixed.capabilities.ibss -e wlan_mgt.fixed.capabilities.cfpoll.ap -e wlan_mgt.fixed.capabilities.privacy -e wlan_mgt.fixed.capabilities.preamble -e wlan_mgt.fixed.capabilities.pbcc -e wlan_mgt.fixed.capabilities.agility -e wlan_mgt.fixed.capabilities.spec_man -e wlan_mgt.fixed.capabilities.short_slot_time -e wlan_mgt.fixed.capabilities.apsd -e wlan_mgt.fixed.capabilities.radio_measurement -e wlan_mgt.fixed.capabilities.dsss_ofdm -e wlan_mgt.fixed.capabilities.del_blk_ack -e wlan_mgt.fixed.capabilities.imm_blk_ack -e wlan_mgt.fixed.listen_ival -e wlan_mgt.fixed.current_ap -e wlan_mgt.fixed.status_code -e wlan_mgt.fixed.timestamp -e wlan_mgt.fixed.beacon -e wlan_mgt.fixed.aid -e wlan_mgt.fixed.reason_code -e wlan_mgt.fixed.auth.alg -e wlan_mgt.fixed.auth_seq -e wlan_mgt.fixed.category_code -e wlan_mgt.fixed.htact -e wlan_mgt.fixed.chanwidth -e wlan_mgt.fixed.fragment -e wlan_mgt.tagged.all -e wlan_mgt.ssid -e wlan_mgt.ds.current_channel -e wlan_mgt.tim.dtim_count -e wlan_mgt.tim.dtim_period -e wlan_mgt.tim.bmapctl.multicast -e wlan_mgt.tim.bmapctl.offset -e wlan_mgt.country_info.environment -e wlan_mgt.rsn.version -e wlan_mgt.rsn.gcs.type -e wlan_mgt.rsn.pcs.count -e wlan_mgt.rsn.akms.count -e wlan_mgt.rsn.akms.type -e wlan_mgt.rsn.capabilities.preauth -e wlan_mgt.rsn.capabilities.no_pairwise -e wlan_mgt.rsn.capabilities.ptksa_replay_counter -e wlan_mgt.rsn.capabilities.gtksa_replay_counter -e wlan_mgt.rsn.capabilities.mfpr -e wlan_mgt.rsn.capabilities.mfpc -e wlan_mgt.rsn.capabilities.peerkey -e wlan_mgt.tcprep.trsmt_pow -e wlan_mgt.tcprep.link_mrg -e wlan.wep.iv -e wlan.wep.key -e wlan.wep.icv -e wlan.tkip.extiv -e wlan.ccmp.extiv -e wlan.qos.tid -e wlan.qos.priority -e wlan.qos.eosp -e wlan.qos.ack -e wlan.qos.amsdupresent -e wlan.qos.buf_state_indicated -e wlan.qos.bit4 -e wlan.qos.txop_dur_req -e wlan.qos.buf_state_indicated -e data.len -E header=y -E separator=, > temp.csv'
		os.system(command)
		X = self.preprocessing('temp.csv', True)
		if self.isKerasModel == True:
			clf = load_model(self.model)
		else:
			clf = joblib.load(self.model)
		prediction = clf.predict(X)
		pred = [i.index(max(i)) for i in prediction]
		chart_info = []
		for i in range(17):
			counter = pred.count(i)
			chart_info.append(counter)
			print(self.classes[i], "{0:.3f}%".format( counter / len(pred)), counter)
		
		self.graph(chart_info)



	def get_attacks_info_csv(self, filename):
		X = self.preprocessing(filename, False)
		clf = joblib.load(self.model)
		prediction = clf.predict(X)
		pred = [i.index(max(i)) for i in prediction]
		chart_info = []
		for i in range(17):
			counter = pred.count(i)
			chart_info.append(counter)
			print(self.classes[i], "{0:.3f}%".format( counter / len(pred)), counter)


		self.graph(chart_info)


