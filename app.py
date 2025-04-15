from datetime import datetime  # Import only datetime, not the entire module
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime
import json
import yfinance as yf
from functools import lru_cache
import time
import tensorflow as tf
from datetime import datetime as dt
from flask import Flask, render_template
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error


app = Flask(__name__)

@app.context_processor
def inject_datetime():
    return {
        'datetime': dt,
        'now': dt.now(),
        'current_year': dt.now().strftime('%Y')
    }
# 📦 @app.context_processor
# This is a Flask decorator that registers a function to inject variables into all Jinja2 templates (HTML files rendered by Flask).

# Think of it as a global context provider for your templates.

# 🔁 def inject_datetime():
# This function returns a dictionary of values that will be available in all your HTML templates, like {{ now }} or {{ current_year }}.


# Indian stock options (NSE tickers with .NS suffix)
STOCK_OPTIONS = {
    # Top 50 Large-Cap Stocks
    "RELIANCE.NS": "Reliance Industries Ltd",
    "TCS.NS": "Tata Consultancy Services Ltd",
    "HDFCBANK.NS": "HDFC Bank Ltd",
    "ICICIBANK.NS": "ICICI Bank Ltd",
    "INFY.NS": "Infosys Ltd",
    "HINDUNILVR.NS": "Hindustan Unilever Ltd",
    "ITC.NS": "ITC Ltd",
    "BHARTIARTL.NS": "Bharti Airtel Ltd",
    "SBIN.NS": "State Bank of India",
    "LICI.NS": "Life Insurance Corporation of India",
    "BAJFINANCE.NS": "Bajaj Finance Ltd",
    "LT.NS": "Larsen & Toubro Ltd",
    "KOTAKBANK.NS": "Kotak Mahindra Bank Ltd",
    "ASIANPAINT.NS": "Asian Paints Ltd",
    "AXISBANK.NS": "Axis Bank Ltd",
    "MARUTI.NS": "Maruti Suzuki India Ltd",
    "SUNPHARMA.NS": "Sun Pharmaceutical Industries Ltd",
    "TITAN.NS": "Titan Company Ltd",
    "ULTRACEMCO.NS": "UltraTech Cement Ltd",
    "NESTLEIND.NS": "Nestle India Ltd",
    "ONGC.NS": "Oil & Natural Gas Corporation Ltd",
    "POWERGRID.NS": "Power Grid Corporation of India Ltd",
    "NTPC.NS": "NTPC Ltd",
    "HCLTECH.NS": "HCL Technologies Ltd",
    "ADANIENT.NS": "Adani Enterprises Ltd",
    "WIPRO.NS": "Wipro Ltd",
    "JSWSTEEL.NS": "JSW Steel Ltd",
    "BAJAJ-AUTO.NS": "Bajaj Auto Ltd",
    "GRASIM.NS": "Grasim Industries Ltd",
    "TATASTEEL.NS": "Tata Steel Ltd",
    "ADANIPORTS.NS": "Adani Ports and Special Economic Zone Ltd",
    "COALINDIA.NS": "Coal India Ltd",
    "M&M.NS": "Mahindra & Mahindra Ltd",
    "TECHM.NS": "Tech Mahindra Ltd",
    "VEDL.NS": "Vedanta Ltd",
    "CIPLA.NS": "Cipla Ltd",
    "BPCL.NS": "Bharat Petroleum Corporation Ltd",
    "INDUSINDBK.NS": "IndusInd Bank Ltd",
    "EICHERMOT.NS": "Eicher Motors Ltd",
    "DRREDDY.NS": "Dr. Reddy's Laboratories Ltd",
    "HEROMOTOCO.NS": "Hero MotoCorp Ltd",
    "DIVISLAB.NS": "Divi's Laboratories Ltd",
    "BRITANNIA.NS": "Britannia Industries Ltd",
    "SHREECEM.NS": "Shree Cement Ltd",
    "HINDALCO.NS": "Hindalco Industries Ltd",
    "APOLLOHOSP.NS": "Apollo Hospitals Enterprise Ltd",
    "BAJAJFINSV.NS": "Bajaj Finserv Ltd",
    "SBILIFE.NS": "SBI Life Insurance Company Ltd",
    "UPL.NS": "UPL Ltd",
    "ZOMATO.NS": "Zomato Ltd",

    # Mid-Cap and Small-Cap Stocks (Sample)
    "DABUR.NS": "Dabur India Ltd",
    "PIDILITIND.NS": "Pidilite Industries Ltd",
    "HAVELLS.NS": "Havells India Ltd",
    "GODREJCP.NS": "Godrej Consumer Products Ltd",
    "AMBUJACEM.NS": "Ambuja Cements Ltd",
    "INDIGO.NS": "InterGlobe Aviation Ltd (IndiGo)",
    "MOTHERSON.NS": "Samvardhana Motherson International Ltd",
    "AUROPHARMA.NS": "Aurobindo Pharma Ltd",
    "PEL.NS": "Piramal Enterprises Ltd",
    "BOSCHLTD.NS": "Bosch Ltd",
    "BERGEPAINT.NS": "Berger Paints India Ltd",
    "BIOCON.NS": "Biocon Ltd",
    "NAUKRI.NS": "Info Edge (India) Ltd",
    "VOLTAS.NS": "Voltas Ltd",
    "ACC.NS": "ACC Ltd",
    "MINDTREE.NS": "Mindtree Ltd",
    "LUPIN.NS": "Lupin Ltd",
    "ABCAPITAL.NS": "Aditya Birla Capital Ltd",
    "RAMCOCEM.NS": "The Ramco Cements Ltd",
    "JUBLFOOD.NS": "Jubilant Foodworks Ltd",
    "TORNTPHARM.NS": "Torrent Pharmaceuticals Ltd",
    "GLENMARK.NS": "Glenmark Pharmaceuticals Ltd",
    "MRF.NS": "MRF Ltd",
    "FORTIS.NS": "Fortis Healthcare Ltd",
    "MANAPPURAM.NS": "Manappuram Finance Ltd",
    "APLLTD.NS": "Alembic Pharmaceuticals Ltd",
    "ESCORTS.NS": "Escorts Kubota Ltd",
    "ALKEM.NS": "Alkem Laboratories Ltd",
    "NHPC.NS": "NHPC Ltd",
    "PFC.NS": "Power Finance Corporation Ltd",
    "RECLTD.NS": "REC Ltd",
    "SUZLON.NS": "Suzlon Energy Ltd",
    "IRCTC.NS": "Indian Railway Catering & Tourism Corporation Ltd",
    "RVNL.NS": "Rail Vikas Nigam Ltd",
    "IRFC.NS": "Indian Railway Finance Corporation Ltd",
    "YESBANK.NS": "Yes Bank Ltd",
    "IDFCFIRSTB.NS": "IDFC First Bank Ltd",
    "BANDHANBNK.NS": "Bandhan Bank Ltd",
    "RBLBANK.NS": "RBL Bank Ltd",
    "FEDERALBNK.NS": "Federal Bank Ltd",
    "PNB.NS": "Punjab National Bank",
    "BANKBARODA.NS": "Bank of Baroda",
    "CANBK.NS": "Canara Bank",
    "UNIONBANK.NS": "Union Bank of India",
    "IOB.NS": "Indian Overseas Bank",
    "CENTRALBK.NS": "Central Bank of India",
    "UCOBANK.NS": "UCO Bank",
    "MAHABANK.NS": "Bank of Maharashtra",
    "J&KBANK.NS": "Jammu & Kashmir Bank Ltd",
    "KTKBANK.NS": "The Karnataka Bank Ltd",
    "SOUTHBANK.NS": "South Indian Bank Ltd",
    "DCBBANK.NS": "DCB Bank Ltd",
    "ESAFSFB.NS": "ESAF Small Finance Bank Ltd",
    "EQUITASBNK.NS": "Equitas Small Finance Bank Ltd",
    "UJJIVANSFB.NS": "Ujjivan Small Finance Bank Ltd",
    "SURYODAY.NS": "Suryoday Small Finance Bank Ltd",
    "FINCABLES.NS": "Finolex Cables Ltd",
    "KAYNES.NS": "Kaynes Technology India Ltd",
    "POLYCAB.NS": "Polycab India Ltd",
    "APARINDS.NS": "Apar Industries Ltd",
    "CUB.NS": "City Union Bank Ltd",
    "LAURUSLABS.NS": "Laurus Labs Ltd",
    "METROBRAND.NS": "Metro Brands Ltd",
    "KALYANKJIL.NS": "Kalyan Jewellers India Ltd",
    "TANLA.NS": "Tanla Platforms Ltd",
    "MAZDOCK.NS": "Mazagon Dock Shipbuilders Ltd",
    "COCHINSHIP.NS": "Cochin Shipyard Ltd",
    "GMDCLTD.NS": "Gujarat Mineral Development Corporation Ltd",
    "HUDCO.NS": "Housing & Urban Development Corporation Ltd",
    "NBCC.NS": "NBCC (India) Ltd",
    "RAINBOW.NS": "Rainbow Children's Medicare Ltd",
    "KRISHNADEF.NS": "Krishna Defence & Allied Industries Ltd",
    "SARDAEN.NS": "Sarda Energy & Minerals Ltd",
    "SOMANYCERA.NS": "Somany Ceramics Ltd",
    "SHILPAMED.NS": "Shilpa Medicare Ltd",
    "TARC.NS": "TARC Ltd",
    "ASHOKA.NS": "Ashoka Buildcon Ltd",
    "PRAJIND.NS": "Praj Industries Ltd",
    "JINDALSAW.NS": "Jindal Saw Ltd",
    "JINDALSTEL.NS": "Jindal Steel & Power Ltd",
    "JSWENERGY.NS": "JSW Energy Ltd",
    "JSWINFRA.NS": "JSW Infrastructure Ltd",
    "JYOTHYLAB.NS": "Jyothy Labs Ltd",
    "KAJARIACER.NS": "Kajaria Ceramics Ltd",
    "KANSAINER.NS": "Kansai Nerolac Paints Ltd",
    "KARURVYSYA.NS": "Karur Vysya Bank Ltd",
    "KEC.NS": "KEC International Ltd",
    "KEI.NS": "KEI Industries Ltd",
    "KIMS.NS": "Krishna Institute of Medical Sciences Ltd",
    "KNRCON.NS": "KNR Constructions Ltd",
    "KPITTECH.NS": "KPIT Technologies Ltd",
    "KPRMILL.NS": "K.P.R. Mill Ltd",
    "KRBL.NS": "KRBL Ltd",
    "KSB.NS": "KSB Ltd",
    "LALPATHLAB.NS": "Dr. Lal PathLabs Ltd",
    "LAXMIMACH.NS": "Lakshmi Machine Works Ltd",
    "LEMONTREE.NS": "Lemon Tree Hotels Ltd",
    "LINDEINDIA.NS": "Linde India Ltd",
    "LLOYDSME.NS": "Lloyds Metals & Energy Ltd",
    "LUXIND.NS": "Lux Industries Ltd",
    "MGL.NS": "Mahanagar Gas Ltd",
    "MOTILALOFS.NS": "Motilal Oswal Financial Services Ltd",
    "MPHASIS.NS": "Mphasis Ltd",
    "NATCOPHARM.NS": "Natco Pharma Ltd",
    "NAVINFLUOR.NS": "Navin Fluorine International Ltd",
    "NCC.NS": "NCC Ltd",
    "NEOGEN.NS": "Neogen Chemicals Ltd",
    "NLCINDIA.NS": "NLC India Ltd",
    "OBEROIRLTY.NS": "Oberoi Realty Ltd",
    "OFSS.NS": "Oracle Financial Services Software Ltd",
    "PAGEIND.NS": "Page Industries Ltd",
    "PERSISTENT.NS": "Persistent Systems Ltd",
    "PETRONET.NS": "Petronet LNG Ltd",
    "PFIZER.NS": "Pfizer Ltd",
    "PHOENIXLTD.NS": "The Phoenix Mills Ltd",
    "PIIND.NS": "PI Industries Ltd",
    "PNBHOUSING.NS": "PNB Housing Finance Ltd",
    "POLYMED.NS": "Poly Medicure Ltd",
    "PRESTIGE.NS": "Prestige Estates Projects Ltd",
    "PRINCEPIPE.NS": "Prince Pipes & Fittings Ltd",
    "QUESS.NS": "Quess Corp Ltd",
    "RADICO.NS": "Radico Khaitan Ltd",
    "RAJESHEXPO.NS": "Rajesh Exports Ltd",
    "RATNAMANI.NS": "Ratnamani Metals & Tubes Ltd",
    "REDINGTON.NS": "Redington India Ltd",
    "RELAXO.NS": "Relaxo Footwears Ltd",
    "RHIM.NS": "RHI Magnesita India Ltd",
    "ROUTE.NS": "ROUTE Mobile Ltd",
    "SAIL.NS": "Steel Authority of India Ltd",
    "SANOFI.NS": "Sanofi India Ltd",
    "SCHAEFFLER.NS": "Schaeffler India Ltd",
    "SEQUENT.NS": "Sequent Scientific Ltd",
    "SHARDACROP.NS": "Sharda Cropchem Ltd",
    "SHRIRAMFIN.NS": "Shriram Finance Ltd",
    "SIEMENS.NS": "Siemens Ltd",
    "SOLARINDS.NS": "Solar Industries India Ltd",
    "SONACOMS.NS": "Sona BLW Precision Forgings Ltd",
    "SPARC.NS": "Sun Pharma Advanced Research Company Ltd",
    "STARHEALTH.NS": "Star Health & Allied Insurance Company Ltd",
    "SUPREMEIND.NS": "Supreme Industries Ltd",
    "SUPPETRO.NS": "Supreme Petrochem Ltd",
    "SUPRAJIT.NS": "Suprajit Engineering Ltd",
    "SUPREME.NS": "Supreme Industries Ltd",
    "SUZLON.NS": "Suzlon Energy Ltd",
    "SYNGENE.NS": "Syngene International Ltd",
    "TATACHEM.NS": "Tata Chemicals Ltd",
    "TATACOMM.NS": "Tata Communications Ltd",
    "TATACONSUM.NS": "Tata Consumer Products Ltd",
    "TATAELXSI.NS": "Tata Elxsi Ltd",
    "TATAMOTORS.NS": "Tata Motors Ltd",
    "TATAPOWER.NS": "Tata Power Company Ltd",
    "TATASTEEL.NS": "Tata Steel Ltd",
    "TCIEXP.NS": "TCI Express Ltd",
    "TCNSBRANDS.NS": "TCNS Clothing Co. Ltd",
    "TEAMLEASE.NS": "TeamLease Services Ltd",
    "TECHNOE.NS": "Techno Electric & Engineering Company Ltd",
    "THERMAX.NS": "Thermax Ltd",
    "THYROCARE.NS": "Thyrocare Technologies Ltd",
    "TIMKEN.NS": "Timken India Ltd",
    "TMB.NS": "Tamilnad Mercantile Bank Ltd",
    "TORNTPOWER.NS": "Torrent Power Ltd",
    "TRENT.NS": "Trent Ltd",
    "TRIDENT.NS": "Trident Ltd",
    "TTKPRESTIG.NS": "TTK Prestige Ltd",
    "TV18BRDCST.NS": "TV18 Broadcast Ltd",
    "TVSMOTOR.NS": "TVS Motor Company Ltd",
    "UBL.NS": "United Breweries Ltd",
    "UJJIVAN.NS": "Ujjivan Financial Services Ltd",
    "ULTRATECH.NS": "UltraTech Cement Ltd",
    "UNOMINDA.NS": "UNO Minda Ltd",
    "VAIBHAVGBL.NS": "Vaibhav Global Ltd",
    "VBL.NS": "Varun Beverages Ltd",
    "VEDL.NS": "Vedanta Ltd",
    "VGUARD.NS": "V-Guard Industries Ltd",
    "VINATIORGA.NS": "Vinati Organics Ltd",
    "VOLTAS.NS": "Voltas Ltd",
    "VTL.NS": "Vardhman Textiles Ltd",
    "WABCOINDIA.NS": "WABCO India Ltd",
    "WELCORP.NS": "Welspun Corp Ltd",
    "WELSPUNIND.NS": "Welspun India Ltd",
    "WESTLIFE.NS": "Westlife Foodworld Ltd",
    "WHIRLPOOL.NS": "Whirlpool of India Ltd",
    "WIPRO.NS": "Wipro Ltd",
    "WOCKPHARMA.NS": "Wockhardt Ltd",
    "ZEEL.NS": "Zee Entertainment Enterprises Ltd",
    "ZENSARTECH.NS": "Zensar Technologies Ltd",
    "ZYDUSLIFE.NS": "Zydus Lifesciences Ltd",
    "ZYDUSWELL.NS": "Zydus Wellness Ltd",
    "3MINDIA.NS": "3M India Ltd",
    "AARTIIND.NS": "Aarti Industries Ltd",
    "ABB.NS": "ABB India Ltd",
    "ABBOTINDIA.NS": "Abbott India Ltd",
    "ADANIGREEN.NS": "Adani Green Energy Ltd",
    "ADANIPORTS.NS": "Adani Ports and Special Economic Zone Ltd",
    "ADANIPOWER.NS": "Adani Power Ltd",
    "ADANITRANS.NS": "Adani Transmission Ltd",
    "ADVENZYMES.NS": "Advanced Enzyme Technologies Ltd",
    "AEGISCHEM.NS": "Aegis Logistics Ltd",
    "AFFLE.NS": "Affle India Ltd",
    "AJANTPHARM.NS": "Ajanta Pharma Ltd",
    "AKZOINDIA.NS": "Akzo Nobel India Ltd",
    "ALKYLAMINE.NS": "Alkyl Amines Chemicals Ltd",
    "ALLCARGO.NS": "Allcargo Logistics Ltd",
    "AMARAJABAT.NS": "Amara Raja Batteries Ltd",
    "AMBER.NS": "Amber Enterprises India Ltd",
    "APOLLOTYRE.NS": "Apollo Tyres Ltd",
    "APTUS.NS": "Aptus Value Housing Finance India Ltd",
    "ASAHIINDIA.NS": "Asahi India Glass Ltd",
    "ASHOKLEY.NS": "Ashok Leyland Ltd",
    "ASTERDM.NS": "Aster DM Healthcare Ltd",
    "ASTRAZEN.NS": "AstraZeneca Pharma India Ltd",
    "ATUL.NS": "Atul Ltd",
    "AUBANK.NS": "AU Small Finance Bank Ltd",
    "AVANTIFEED.NS": "Avanti Feeds Ltd",
    "AVTNPL.NS": "AVT Natural Products Ltd",
    "BAJAJELEC.NS": "Bajaj Electricals Ltd",
    "BAJAJHLDNG.NS": "Bajaj Holdings & Investment Ltd",
    "BALAMINES.NS": "Balaji Amines Ltd",
    "BALKRISIND.NS": "Balkrishna Industries Ltd",
    "BALRAMCHIN.NS": "Balrampur Chini Mills Ltd",
    "BANARISUG.NS": "Bannari Amman Sugars Ltd",
    "BATAINDIA.NS": "Bata India Ltd",
    "BBTC.NS": "Bombay Burmah Trading Corporation Ltd",
    "BDL.NS": "Bharat Dynamics Ltd",
    "BEL.NS": "Bharat Electronics Ltd",
    "BEML.NS": "BEML Ltd",
    "BHARATFORG.NS": "Bharat Forge Ltd",
    "BHARATRAS.NS": "Bharat Rasayan Ltd",
    "BHEL.NS": "Bharat Heavy Electricals Ltd",
    "BLUEDART.NS": "Blue Dart Express Ltd",
    "BLUESTARCO.NS": "Blue Star Ltd",
    "BORORENEW.NS": "Borosil Renewables Ltd",
    "BOSCH.NS": "Bosch Ltd",
    "BRIGADE.NS": "Brigade Enterprises Ltd",
    "BSE.NS": "BSE Ltd",
    "BSOFT.NS": "Birlasoft Ltd",
    "CADILAHC.NS": "Cadila Healthcare Ltd",
    "CANFINHOME.NS": "Can Fin Homes Ltd",
    "CASTROLIND.NS": "Castrol India Ltd",
    "CCL.NS": "CCL Products India Ltd",
    "CDSL.NS": "Central Depository Services India Ltd",
    "CEATLTD.NS": "CEAT Ltd",
    "CENTURYPLY.NS": "Century Plyboards India Ltd",
    "CENTURYTEX.NS": "Century Textiles & Industries Ltd",
    "CERA.NS": "Cera Sanitaryware Ltd",
    "CHALET.NS": "Chalet Hotels Ltd",
    "CHAMBLFERT.NS": "Chambal Fertilizers & Chemicals Ltd",
    "CHEMPLAST.NS": "Chemplast Sanmar Ltd",
    "CHOLAHLDNG.NS": "Cholamandalam Financial Holdings Ltd",
    "CIPLA.NS": "Cipla Ltd",
    "CLEAN.NS": "Clean Science and Technology Ltd",
    "COFORGE.NS": "Coforge Ltd",
    "CONCOR.NS": "Container Corporation of India Ltd",
    "COROMANDEL.NS": "Coromandel International Ltd",
    "CREDITACC.NS": "CreditAccess Grameen Ltd",
    "CRISIL.NS": "CRISIL Ltd",
    "CROMPTON.NS": "Crompton Greaves Consumer Electricals Ltd",
    "CSBBANK.NS": "CSB Bank Ltd",
    "CUMMINSIND.NS": "Cummins India Ltd",
    "CYIENT.NS": "Cyient Ltd",
    "DALBHARAT.NS": "Dalmia Bharat Ltd",
    "DCMSHRIRAM.NS": "DCM Shriram Ltd",
    "DEEPAKNTR.NS": "Deepak Nitrite Ltd",
    "DELHIVERY.NS": "Delhivery Ltd",
    "DEVYANI.NS": "Devyani International Ltd",
    "DIXON.NS": "Dixon Technologies India Ltd",
    "DLF.NS": "DLF Ltd",
    "DMART.NS": "Avenue Supermarts Ltd (D-Mart)",
    "DRREDDY.NS": "Dr. Reddy's Laboratories Ltd",
    "DWARKESH.NS": "Dwarikesh Sugar Industries Ltd",
    "ECLERX.NS": "eClerx Services Ltd",
    "EDELWEISS.NS": "Edelweiss Financial Services Ltd",
    "EIDPARRY.NS": "EID Parry India Ltd",
    "EIHOTEL.NS": "EIH Ltd",
    "ELGIEQUIP.NS": "Elgi Equipments Ltd",
    "EMAMILTD.NS": "Emami Ltd",
    "ENDURANCE.NS": "Endurance Technologies Ltd",
    "ENGINERSIN.NS": "Engineers India Ltd",
    "ERIS.NS": "Eris Lifesciences Ltd",
    "EXIDEIND.NS": "Exide Industries Ltd",
    "FACT.NS": "Fertilizers and Chemicals Travancore Ltd",
    "FDC.NS": "FDC Ltd",
    "FINEORG.NS": "Fine Organic Industries Ltd",
    "FINPIPE.NS": "Finolex Industries Ltd",
    "FLUOROCHEM.NS": "Gujarat Fluorochemicals Ltd",
    "FORTIS.NS": "Fortis Healthcare Ltd",
    "FSL.NS": "Firstsource Solutions Ltd",
    "GALAXYSURF.NS": "Galaxy Surfactants Ltd",
    "GARFIBRES.NS": "Garware Technical Fibres Ltd",
    "GMMPFAUDLR.NS": "GMM Pfaudler Ltd",
    "GNFC.NS": "Gujarat Narmada Valley Fertilizers & Chemicals Ltd",
    "GODFRYPHLP.NS": "Godfrey Phillips India Ltd",
    "GODREJAGRO.NS": "Godrej Agrovet Ltd",
    "GODREJIND.NS": "Godrej Industries Ltd",
    "GODREJPROP.NS": "Godrej Properties Ltd",
    "GRANULES.NS": "Granules India Ltd",
    "GRAPHITE.NS": "Graphite India Ltd",
    "GRINDWELL.NS": "Grindwell Norton Ltd",
    "GSFC.NS": "Gujarat State Fertilizers & Chemicals Ltd",
    "GSPL.NS": "Gujarat State Petronet Ltd",
    "GUJGASLTD.NS": "Gujarat Gas Ltd",
    "HAL.NS": "Hindustan Aeronautics Ltd",
    "HATHWAY.NS": "Hathway Cable & Datacom Ltd",
    "HATSUN.NS": "Hatsun Agro Product Ltd",
    "HEG.NS": "HEG Ltd",
    "HEIDELBERG.NS": "HeidelbergCement India Ltd",
    "HFCL.NS": "HFCL Ltd",
    "HGINFRA.NS": "H.G. Infra Engineering Ltd",
    "HIKAL.NS": "Hikal Ltd",
    "HIMATSEIDE.NS": "Himatsingka Seide Ltd",
    "HINDCOPPER.NS": "Hindustan Copper Ltd",
    "HINDZINC.NS": "Hindustan Zinc Ltd",
    "HONAUT.NS": "Honeywell Automation India Ltd",
    "HONDAPOWER.NS": "Honda India Power Products Ltd",
    "HUDCO.NS": "Housing & Urban Development Corporation Ltd",
    "IBREALEST.NS": "Indiabulls Real Estate Ltd",
    "IBULHSGFIN.NS": "Indiabulls Housing Finance Ltd",
    "ICICIPRULI.NS": "ICICI Prudential Life Insurance Company Ltd",
    "ICIL.NS": "Indo Count Industries Ltd",
    "ICRA.NS": "ICRA Ltd",
    "IDBI.NS": "IDBI Bank Ltd",
    "IDEA.NS": "Vodafone Idea Ltd",
    "IEX.NS": "Indian Energy Exchange Ltd",
    "IFBIND.NS": "IFB Industries Ltd",
    "IGL.NS": "Indraprastha Gas Ltd",
    "IIFL.NS": "IIFL Finance Ltd",
    "IIFLWAM.NS": "IIFL Wealth Management Ltd",
    "INDIACEM.NS": "India Cements Ltd",
    "INDIAMART.NS": "Indiamart Intermesh Ltd",
    "INDIANB.NS": "Indian Bank",
    "INDIGO.NS": "InterGlobe Aviation Ltd (IndiGo)",
    "INDIGOPNTS.NS": "Indigo Paints Ltd",
    "INDOCO.NS": "Indoco Remedies Ltd",
    "INDUSINDBK.NS": "IndusInd Bank Ltd",
    "INOXLEISUR.NS": "INOX Leisure Ltd",
    "INOXWIND.NS": "Inox Wind Ltd",
    "IOB.NS": "Indian Overseas Bank",
    "IOLCP.NS": "IOL Chemicals and Pharmaceuticals Ltd",
    "IPCALAB.NS": "IPCA Laboratories Ltd",
    "IRB.NS": "IRB Infrastructure Developers Ltd",
    "ISEC.NS": "ICICI Securities Ltd",
    "ITDC.NS": "India Tourism Development Corporation Ltd",
    "ITI.NS": "ITI Ltd",
    "IVC.NS": "IL&FS Investment Managers Ltd",
    "JBCHEPHARM.NS": "JB Chemicals & Pharmaceuticals Ltd",
    "JINDALSAW.NS": "Jindal Saw Ltd",
    "JKCEMENT.NS": "JK Cement Ltd",
    "JKLAKSHMI.NS": "JK Lakshmi Cement Ltd",
    "JKPAPER.NS": "JK Paper Ltd",
    "JKTYRE.NS": "JK Tyre & Industries Ltd",
    "JMFINANCIL.NS": "JM Financial Ltd",
    "JSL.NS": "Jindal Stainless Ltd",
    "JSWISPAT.NS": "JSW Ispat Special Products Ltd",
    "JUBLINGREA.NS": "Jubilant Ingrevia Ltd",
    "JUBLPHARMA.NS": "Jubilant Pharmova Ltd",
    "JUSTDIAL.NS": "Just Dial Ltd",
    "JWL.NS": "Jupiter Wagons Ltd",
    "KAJARIACER.NS": "Kajaria Ceramics Ltd",
    "KALPATPOWR.NS": "Kalpataru Power Transmission Ltd",
    "KANSAINER.NS": "Kansai Nerolac Paints Ltd",
    "KARURVYSYA.NS": "Karur Vysya Bank Ltd",
    "KAYNES.NS": "Kaynes Technology India Ltd",
    "KEC.NS": "KEC International Ltd",
    "KEI.NS": "KEI Industries Ltd",
    "KIMS.NS": "Krishna Institute of Medical Sciences Ltd",
    "KNRCON.NS": "KNR Constructions Ltd",
    "KPITTECH.NS": "KPIT Technologies Ltd",
    "KPRMILL.NS": "K.P.R. Mill Ltd",
    "KRBL.NS": "KRBL Ltd",
    "KSB.NS": "KSB Ltd",
    "LALPATHLAB.NS": "Dr. Lal PathLabs Ltd",
    "LAXMIMACH.NS": "Lakshmi Machine Works Ltd",
    "LEMONTREE.NS": "Lemon Tree Hotels Ltd",
    "LINDEINDIA.NS": "Linde India Ltd",
    "LLOYDSME.NS": "Lloyds Metals & Energy Ltd",
    "LUXIND.NS": "Lux Industries Ltd",
    "M&M.NS": "Mahindra & Mahindra Ltd",
    "M&MFIN.NS": "Mahindra & Mahindra Financial Services Ltd",
    "MGL.NS": "Mahanagar Gas Ltd",
    "MOTILALOFS.NS": "Motilal Oswal Financial Services Ltd",
    "MPHASIS.NS": "Mphasis Ltd",
    "MRPL.NS": "Mangalore Refinery and Petrochemicals Ltd",
    "MUTHOOTFIN.NS": "Muthoot Finance Ltd",
    "NATCOPHARM.NS": "Natco Pharma Ltd",
    "NAUKRI.NS": "Info Edge (India) Ltd",
    "NAVINFLUOR.NS": "Navin Fluorine International Ltd",
    "NCC.NS": "NCC Ltd",
    "NESCO.NS": "Nesco Ltd",
    "NESTLEIND.NS": "Nestle India Ltd",
    "NETWORK18.NS": "Network18 Media & Investments Ltd",
    "NFL.NS": "National Fertilizers Ltd",
    "NH.NS": "Narayana Hrudayalaya Ltd",
    "NHPC.NS": "NHPC Ltd",
    "NILKAMAL.NS": "Nilkamal Ltd",
    "NLCINDIA.NS": "NLC India Ltd",
    "NMDC.NS": "NMDC Ltd",
    "NOCIL.NS": "NOCIL Ltd",
    "NRBBEARING.NS": "NRB Bearing Ltd",
    "NUVOCO.NS": "Nuvoco Vistas Corporation Ltd",
    "OBEROIRLTY.NS": "Oberoi Realty Ltd",
    "OFSS.NS": "Oracle Financial Services Software Ltd",
    "OIL.NS": "Oil India Ltd",
    "OLECTRA.NS": "Olectra Greentech Ltd",
    "OMAXE.NS": "Omaxe Ltd",
    "ONGC.NS": "Oil & Natural Gas Corporation Ltd",
    "ORIENTELEC.NS": "Orient Electric Ltd",
    "PAGEIND.NS": "Page Industries Ltd",
    "PEL.NS": "Piramal Enterprises Ltd",
    "PERSISTENT.NS": "Persistent Systems Ltd",
    "PETRONET.NS": "Petronet LNG Ltd",
    "PFIZER.NS": "Pfizer Ltd",
    "PGHL.NS": "Procter & Gamble Health Ltd",
    "PHOENIXLTD.NS": "The Phoenix Mills Ltd",
    "PIDILITIND.NS": "Pidilite Industries Ltd",
    "PIIND.NS": "PI Industries Ltd",
    "PNBHOUSING.NS": "PNB Housing Finance Ltd",
    "POLYCAB.NS": "Polycab India Ltd",
    "POLYMED.NS": "Poly Medicure Ltd",
    "POWERINDIA.NS": "Hitachi Energy India Ltd",
    "PRAJIND.NS": "Praj Industries Ltd",
    "PRESTIGE.NS": "Prestige Estates Projects Ltd",
    "PRINCEPIPE.NS": "Prince Pipes & Fittings Ltd",
    "PRIVISCL.NS": "Privi Speciality Chemicals Ltd",
    "PROZONINTU.NS": "Prozone Intu Properties Ltd",
    "PSB.NS": "Punjab & Sind Bank",
    "PSPPROJECT.NS": "PSP Projects Ltd",
    "PTC.NS": "PTC India Ltd",
    "PVRINOX.NS": "PVR Inox Ltd",
    "QUESS.NS": "Quess Corp Ltd",
    "RADICO.NS": "Radico Khaitan Ltd",
    "RAIN.NS": "Rain Industries Ltd",
    "RAJESHEXPO.NS": "Rajesh Exports Ltd",
    "RALLIS.NS": "Rallis India Ltd",
    "RATNAMANI.NS": "Ratnamani Metals & Tubes Ltd",
    "RAYMOND.NS": "Raymond Ltd",
    "RBLBANK.NS": "RBL Bank Ltd",
    "REDINGTON.NS": "Redington India Ltd",
    "RELAXO.NS": "Relaxo Footwears Ltd",
    "RHIM.NS": "RHI Magnesita India Ltd",
    "RITES.NS": "RITES Ltd",
    "ROUTE.NS": "ROUTE Mobile Ltd",
    "RPOWER.NS": "Reliance Power Ltd",
    "RTNINDIA.NS": "RattanIndia Power Ltd",
    "SAIL.NS": "Steel Authority of India Ltd",
    "SANOFI.NS": "Sanofi India Ltd",
    "SARDAEN.NS": "Sarda Energy & Minerals Ltd",
    "SCHAEFFLER.NS": "Schaeffler India Ltd",
    "SCHNEIDER.NS": "Schneider Electric Infrastructure Ltd",
    "SEQUENT.NS": "Sequent Scientific Ltd",
    "SHARDACROP.NS": "Sharda Cropchem Ltd",
    "SHILPAMED.NS": "Shilpa Medicare Ltd",
    "SHOPERSTOP.NS": "Shoppers Stop Ltd",
    "SHRIRAMFIN.NS": "Shriram Finance Ltd",
    "SIEMENS.NS": "Siemens Ltd",
    "SOBHA.NS": "Sobha Ltd",
    "SOLARA.NS": "Solara Active Pharma Sciences Ltd",
    "SOLARINDS.NS": "Solar Industries India Ltd",
    "SONACOMS.NS": "Sona BLW Precision Forgings Ltd",
    "SPANDANA.NS": "Spandana Sphoorty Financial Ltd",
    "SPARC.NS": "Sun Pharma Advanced Research Company Ltd",
    "STAR.NS": "Strides Pharma Science Ltd",
    "STARCEMENT.NS": "Star Cement Ltd",
    "STERTOOLS.NS": "Sterling Tools Ltd",
    "SUDARSCHEM.NS": "Sudarshan Chemical Industries Ltd",
    "SUNDARMFIN.NS": "Sundaram Finance Ltd",
    "SUNDRMFAST.NS": "Sundram Fasteners Ltd",
    "SUNTV.NS": "Sun TV Network Ltd",
    "SUPREMEIND.NS": "Supreme Industries Ltd",
    "SUPRIYA.NS": "Supriya Lifescience Ltd",
    "SUZLON.NS": "Suzlon Energy Ltd",
    "SWANENERGY.NS": "Swan Energy Ltd",
    "SYMPHONY.NS": "Symphony Ltd",
    "SYNGENE.NS": "Syngene International Ltd",
    "TAKE.NS": "Take Solutions Ltd",
    "TANLA.NS": "Tanla Platforms Ltd",
    "TATACHEM.NS": "Tata Chemicals Ltd",
    "TATACOFFEE.NS": "Tata Coffee Ltd",
    "TATACOMM.NS": "Tata Communications Ltd",
    "TATACONSUM.NS": "Tata Consumer Products Ltd",
    "TATAELXSI.NS": "Tata Elxsi Ltd",
    "TATAINVEST.NS": "Tata Investment Corporation Ltd",
    "TATAMETALI.NS": "Tata Metaliks Ltd",
    "TATAMOTORS.NS": "Tata Motors Ltd",
    "TATAPOWER.NS": "Tata Power Company Ltd",
    "TATASTEEL.NS": "Tata Steel Ltd",
    "TATVA.NS": "Tatva Chintan Pharma Chem Ltd",
    "TCIEXP.NS": "TCI Express Ltd",
    "TCNSBRANDS.NS": "TCNS Clothing Co. Ltd",
    "TEAMLEASE.NS": "TeamLease Services Ltd",
    "TECHM.NS": "Tech Mahindra Ltd",
    "THERMAX.NS": "Thermax Ltd",
    "THYROCARE.NS": "Thyrocare Technologies Ltd",
    "TIMKEN.NS": "Timken India Ltd",
    "TITAN.NS": "Titan Company Ltd",
    "TORNTPHARM.NS": "Torrent Pharmaceuticals Ltd",
    "TORNTPOWER.NS": "Torrent Power Ltd",
    "TRENT.NS": "Trent Ltd",
    "TRIDENT.NS": "Trident Ltd",
    "TTKPRESTIG.NS": "TTK Prestige Ltd",
    "TV18BRDCST.NS": "TV18 Broadcast Ltd",
    "TVSMOTOR.NS": "TVS Motor Company Ltd",
    "UBL.NS": "United Breweries Ltd",
    "UCOBANK.NS": "UCO Bank",
    "UFLEX.NS": "UFLEX Ltd",
    "UJJIVAN.NS": "Ujjivan Financial Services Ltd",
    "UJJIVANSFB.NS": "Ujjivan Small Finance Bank Ltd",
    "ULTRACEMCO.NS": "UltraTech Cement Ltd",
    "UNIONBANK.NS": "Union Bank of India",
    "UNOMINDA.NS": "UNO Minda Ltd",
    "UPL.NS": "UPL Ltd",
    "VAIBHAVGBL.NS": "Vaibhav Global Ltd",
    "VAKRANGEE.NS": "Vakrangee Ltd",
    "VBL.NS": "Varun Beverages Ltd",
    "VEDL.NS": "Vedanta Ltd",
    "VESUVIUS.NS": "Vesuvius India Ltd",
    "VGUARD.NS": "V-Guard Industries Ltd",
    "VINATIORGA.NS": "Vinati Organics Ltd",
    "VIPIND.NS": "VIP Industries Ltd",
    "VOLTAS.NS": "Voltas Ltd",
    "VSTIND.NS": "VST Industries Ltd",
    "WABCOINDIA.NS": "WABCO India Ltd",
    "WELCORP.NS": "Welspun Corp Ltd",
    "WELSPUNIND.NS": "Welspun India Ltd",
    "WESTLIFE.NS": "Westlife Foodworld Ltd",
    "WHIRLPOOL.NS": "Whirlpool of India Ltd",
    "WIPRO.NS": "Wipro Ltd",
    "WOCKPHARMA.NS": "Wockhardt Ltd",
    "YESBANK.NS": "Yes Bank Ltd",
    "ZENSARTECH.NS": "Zensar Technologies Ltd",
    "ZEEL.NS": "Zee Entertainment Enterprises Ltd",
    "ZOMATO.NS": "Zomato Ltd",
    "ZYDUSLIFE.NS": "Zydus Lifesciences Ltd",
    "ZYDUSWELL.NS": "Zydus Wellness Ltd"
}
# Cache with 15-minute expiration
def get_ttl_hash(seconds=900):
    return round(time.time() / seconds)

@lru_cache(maxsize=32)
def cached_stock_data(ticker, ttl_hash=None):
    del ttl_hash  # To make time-based cache work
    return get_stock_data(ticker)

# def get_stock_data(ticker):
#     """Fetch stock data using yfinance with multiple fallback attempts"""
#     attempts = [
#         {'period': '1mo', 'interval': '1d'},
#         {'period': '3mo', 'interval': '1d'},
#         {'period': '7d', 'interval': '1h'},
#         {'period': '5d', 'interval': '15m'}
#     ]
    
#     for attempt in attempts:
#         try:
#             data = yf.Ticker(ticker).history(
#                 period=attempt['period'],
#                 interval=attempt['interval']
#             )['Close']
#             if len(data) >= 60:
#                 return data
#         except Exception as e:
#             print(f"Attempt failed ({attempt}): {e}")
    
#     return pd.Series()
def get_stock_data(ticker):
    """Fetch stock data using yfinance with fallback attempts and headers"""
    import requests
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})

    attempts = [
        {'period': '1mo', 'interval': '1d'},
        {'period': '3mo', 'interval': '1d'},
        {'period': '7d', 'interval': '1h'},
        {'period': '5d', 'interval': '15m'}
    ]

    for attempt in attempts:
        try:
            stock = yf.Ticker(ticker, session=session)
            data = stock.history(period=attempt['period'], interval=attempt['interval'])['Close']
            if len(data) >= 60:
                return data
        except Exception as e:
            print(f"⚠️ Attempt failed for {ticker} with {attempt}: {e}")

    print(f"❌ No data returned for {ticker}")
    return pd.Series()

def get_stock_info(ticker):
    """Get sector, previous close price, and company website"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'sector': info.get('sector', 'N/A'),
            'current_price': info.get('previousClose', 0),
            'website': info.get('website', '#'),
            'logo_url': info.get('logo_url', '')
        }
    except:
        return {'sector': 'N/A', 'current_price': 0, 'website': '#', 'logo_url': ''}

def advanced_stock_analysis(ticker):
    """Comprehensive stock analysis with multiple indicators"""
    try:
        # Fetch historical stock data
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period='1y')
        
        # Basic Price Analysis
        current_price = hist_data['Close'][-1]
        avg_price_52_weeks_high = hist_data['High'].max()
        avg_price_52_weeks_low = hist_data['Low'].min()
        price_distance_from_52_week_high = ((current_price - avg_price_52_weeks_low) / 
                                            (avg_price_52_weeks_high - avg_price_52_weeks_low)) * 100
        
        # Volume Analysis
        avg_volume = hist_data['Volume'].mean()
        current_volume = hist_data['Volume'][-1]
        volume_change = ((current_volume - avg_volume) / avg_volume) * 100
        
        # Technical Indicators
        # Simple Moving Averages
        ma_20 = hist_data['Close'].rolling(window=20).mean()[-1]
        ma_50 = hist_data['Close'].rolling(window=50).mean()[-1]
        ma_200 = hist_data['Close'].rolling(window=200).mean()[-1]
        
        # Relative Strength Index (RSI)
        def calculate_rsi(data, periods=14):
            delta = data.diff()
            dUp, dDown = delta.copy(), delta.copy()
            dUp[dUp < 0] = 0
            dDown[dDown > 0] = 0
            
            RollUp = dUp.rolling(window=periods).mean()
            RollDown = dDown.abs().rolling(window=periods).mean()
            
            RS = RollUp / RollDown
            RSI = 100.0 - (100.0 / (1.0 + RS))
            return RSI
        
        rsi = calculate_rsi(hist_data['Close'])[-1]
        
        # Decision Logic
        decision_score = 0
        decision_reasons = []
        
        # Price Trend Analysis
        if current_price > ma_20 and current_price > ma_50 and current_price > ma_200:
            decision_score += 2
            decision_reasons.append("Strong Upward Price Trend")
        elif current_price < ma_20 and current_price < ma_50 and current_price < ma_200:
            decision_score -= 2
            decision_reasons.append("Weak Downward Price Trend")
        
        # Volume Analysis
        if current_volume > avg_volume * 1.5:
            decision_score += 1
            decision_reasons.append("High Trading Volume")
        
        # RSI Analysis
        if rsi < 30:
            decision_score += 2
            decision_reasons.append("Oversold Condition (Potential Buy Signal)")
        elif rsi > 70:
            decision_score -= 2
            decision_reasons.append("Overbought Condition (Potential Sell Signal)")
        
        # Price Positioning
        if price_distance_from_52_week_high < 20:
            decision_score += 1
            decision_reasons.append("Close to 52-Week High")
        elif price_distance_from_52_week_high > 80:
            decision_score -= 1
            decision_reasons.append("Far from 52-Week High")
        
        # Final Recommendation
        if decision_score > 2:
            recommendation = "Strong Buy"
            recommendation_color = "green"
        elif decision_score > 0:
            recommendation = "Moderate Buy"
            recommendation_color = "Darkgreen"
        elif decision_score == 0:
            recommendation = "Neutral"
            recommendation_color = "Darkorange"
        elif decision_score > -2:
            recommendation = "Moderate Sell"
            recommendation_color = "Lightred"
        else:
            recommendation = "Strong Sell"
            recommendation_color = "red"
        
        # Financial Fundamentals
        try:
            financials = {
                'Market Cap': stock.info.get('marketCap', 'N/A'),
                'P/E Ratio': stock.info.get('trailingPE', 'N/A'),
                'Dividend Yield': stock.info.get('dividendYield', 'N/A'),
                'EPS': stock.info.get('trailingEps', 'N/A'),
                'Beta': stock.info.get('beta', 'N/A')
            }
        except:
            financials = {}
        
        return {
            'current_price': current_price,
            '52_week_high': avg_price_52_weeks_high,
            '52_week_low': avg_price_52_weeks_low,
            'price_distance_from_52week_high': price_distance_from_52_week_high,
            'ma_20': ma_20,
            'ma_50': ma_50,
            'ma_200': ma_200,
            'rsi': rsi,
            'volume_change': volume_change,
            'recommendation': recommendation,
            'recommendation_color': recommendation_color,
            'decision_reasons': decision_reasons,
            'financials': financials
        }
    
    except Exception as e:
        print(f"Advanced analysis error: {e}")
        return None

def prepare_data(data, look_back=60):
    """Prepare data for LSTM model"""
    data_values = data.values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_test, y_test, scaler

def create_lstm_model(input_shape):
    """Create LSTM model with improved architecture"""
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(units=100, return_sequences=True),
        Dropout(0.3),
        LSTM(units=50),
        Dropout(0.3),
        Dense(units=25, activation='relu'),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# def get_live_price(ticker):
#     """Get live stock price"""
#     try:
#         stock = yf.Ticker(ticker)
#         live_price = stock.history(period='1d')['Close'].iloc[-1]
#         return float(live_price)
#     except Exception as e:
#         print(f"Error getting live price: {e}")
#         return None
def get_live_price(ticker):
    """Get live stock price with fallback for empty history"""
    try:
        import requests
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})

        stock = yf.Ticker(ticker, session=session)
        hist = stock.history(period='1d')

        if hist.empty:
            raise ValueError("Live price data is empty.")

        live_price = hist['Close'].iloc[-1]
        return float(live_price)
    except Exception as e:
        print(f"❌ Error getting live price for {ticker}: {e}")
        return None
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

def predict_stock_movement(ticker):
    """Main prediction function with error handling, live price, and advanced analysis"""
    try:
        # Get live price first
        live_price = get_live_price(ticker)
        
        # Perform advanced stock analysis
        advanced_analysis = advanced_stock_analysis(ticker)
        
        stock_prices = cached_stock_data(ticker, get_ttl_hash())
        
        if stock_prices.empty:
            raise ValueError("No data available for this stock")
        if len(stock_prices) < 60:
            raise ValueError(f"Only {len(stock_prices)} data points available (need 60)")
        
        # Prepare data and train model
        X_train, y_train, _, _, scaler = prepare_data(stock_prices)
        model = create_lstm_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
        
        # Prepare last 60 data points for prediction
        last_60 = stock_prices[-60:].values.astype('float32')
        last_60_scaled = scaler.transform(last_60.reshape(-1, 1))
        X_pred = np.array([last_60_scaled.flatten()])
        X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))
        
        # Generate predictions
        predictions = []
        for _ in range(100):
            pred = model.predict(X_pred, verbose=0)
            predictions.append(scaler.inverse_transform(pred)[0][0])
        
        predicted_value = np.mean(predictions)
        last_price = stock_prices.iloc[-1]
        
        movement = "Up" if predicted_value > last_price else "Down"
        confidence = abs((predicted_value - last_price) / last_price) * 100
        
        # Ensure predicted_value is a float
        predicted_value = float(predicted_value)
        
        # Prepare chart data
        dates = stock_prices.index[-60:].strftime('%Y-%m-%d').tolist()
        prices = stock_prices[-60:].values.tolist()
        chart_data = (dates, prices)
        
        # Get stock info
        stock_info = get_stock_info(ticker)
        
        return (movement, confidence, predicted_value, last_price,
                chart_data, stock_info, live_price, advanced_analysis)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        stock_info = get_stock_info(ticker)
        return (f"Error: {str(e)}", 0.0, stock_info['current_price'], 
                stock_info['current_price'], ([], []), stock_info, None, None)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        ticker = request.form['ticker']
        stock_name = STOCK_OPTIONS.get(ticker, ticker)
        
        result = predict_stock_movement(ticker)
        (movement, confidence, predicted_value, last_price,
         chart_data, stock_info, live_price, advanced_analysis) = result
        
        return render_template('result.html',
                            ticker=stock_name,
                            movement=movement,
                            confidence=f"{confidence:.2f}" if isinstance(confidence, float) else "N/A",
                            predicted_value=f"{predicted_value:.2f}" if isinstance(predicted_value, float) else "N/A",
                            last_price=f"{last_price:.2f}" if isinstance(last_price, float) else "N/A",
                            live_price=f"{live_price:.2f}" if live_price else "N/A",
                            chart_data=json.dumps(chart_data),
                            sector=stock_info['sector'],
                            current_price=f"{stock_info['current_price']:.2f}" if isinstance(stock_info['current_price'], float) else "N/A",
                            website=stock_info['website'],
                            logo_url=stock_info['logo_url'],
                            timestamp=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            advanced_analysis=advanced_analysis)
    
    return render_template('index.html', stock_options=STOCK_OPTIONS)

if __name__ == '__main__':
    app.run(debug=True)