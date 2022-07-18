%% Import data from text file
% Script for importing data from the following text file:
%
%    filename: C:\Users\Rob Roy\OneDrive - Maynooth University\Documents\MATLAB\4dsigma1new_0_24std_BLUR_spectrums_valid.csv
%
% Auto-generated by MATLAB on 05-Jul-2022 15:36:05

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 1001);

% Specify range and delimiter
opts.DataLines = [2, 20001];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["Var1", "VarName2", "VarName3", "VarName4", "VarName5", "VarName6", "VarName7", "VarName8", "VarName9", "VarName10", "VarName11", "VarName12", "VarName13", "VarName14", "VarName15", "VarName16", "VarName17", "VarName18", "VarName19", "VarName20", "VarName21", "VarName22", "VarName23", "VarName24", "VarName25", "VarName26", "VarName27", "VarName28", "VarName29", "VarName30", "VarName31", "VarName32", "VarName33", "VarName34", "VarName35", "VarName36", "VarName37", "VarName38", "VarName39", "VarName40", "VarName41", "VarName42", "VarName43", "VarName44", "VarName45", "VarName46", "VarName47", "VarName48", "VarName49", "VarName50", "VarName51", "VarName52", "VarName53", "VarName54", "VarName55", "VarName56", "VarName57", "VarName58", "VarName59", "VarName60", "VarName61", "VarName62", "VarName63", "VarName64", "VarName65", "VarName66", "VarName67", "VarName68", "VarName69", "VarName70", "VarName71", "VarName72", "VarName73", "VarName74", "VarName75", "VarName76", "VarName77", "VarName78", "VarName79", "VarName80", "VarName81", "VarName82", "VarName83", "VarName84", "VarName85", "VarName86", "VarName87", "VarName88", "VarName89", "VarName90", "VarName91", "VarName92", "VarName93", "VarName94", "VarName95", "VarName96", "VarName97", "VarName98", "VarName99", "VarName100", "VarName101", "VarName102", "VarName103", "VarName104", "VarName105", "VarName106", "VarName107", "VarName108", "VarName109", "VarName110", "VarName111", "VarName112", "VarName113", "VarName114", "VarName115", "VarName116", "VarName117", "VarName118", "VarName119", "VarName120", "VarName121", "VarName122", "VarName123", "VarName124", "VarName125", "VarName126", "VarName127", "VarName128", "VarName129", "VarName130", "VarName131", "VarName132", "VarName133", "VarName134", "VarName135", "VarName136", "VarName137", "VarName138", "VarName139", "VarName140", "VarName141", "VarName142", "VarName143", "VarName144", "VarName145", "VarName146", "VarName147", "VarName148", "VarName149", "VarName150", "VarName151", "VarName152", "VarName153", "VarName154", "VarName155", "VarName156", "VarName157", "VarName158", "VarName159", "VarName160", "VarName161", "VarName162", "VarName163", "VarName164", "VarName165", "VarName166", "VarName167", "VarName168", "VarName169", "VarName170", "VarName171", "VarName172", "VarName173", "VarName174", "VarName175", "VarName176", "VarName177", "VarName178", "VarName179", "VarName180", "VarName181", "VarName182", "VarName183", "VarName184", "VarName185", "VarName186", "VarName187", "VarName188", "VarName189", "VarName190", "VarName191", "VarName192", "VarName193", "VarName194", "VarName195", "VarName196", "VarName197", "VarName198", "VarName199", "VarName200", "VarName201", "VarName202", "VarName203", "VarName204", "VarName205", "VarName206", "VarName207", "VarName208", "VarName209", "VarName210", "VarName211", "VarName212", "VarName213", "VarName214", "VarName215", "VarName216", "VarName217", "VarName218", "VarName219", "VarName220", "VarName221", "VarName222", "VarName223", "VarName224", "VarName225", "VarName226", "VarName227", "VarName228", "VarName229", "VarName230", "VarName231", "VarName232", "VarName233", "VarName234", "VarName235", "VarName236", "VarName237", "VarName238", "VarName239", "VarName240", "VarName241", "VarName242", "VarName243", "VarName244", "VarName245", "VarName246", "VarName247", "VarName248", "VarName249", "VarName250", "VarName251", "VarName252", "VarName253", "VarName254", "VarName255", "VarName256", "VarName257", "VarName258", "VarName259", "VarName260", "VarName261", "VarName262", "VarName263", "VarName264", "VarName265", "VarName266", "VarName267", "VarName268", "VarName269", "VarName270", "VarName271", "VarName272", "VarName273", "VarName274", "VarName275", "VarName276", "VarName277", "VarName278", "VarName279", "VarName280", "VarName281", "VarName282", "VarName283", "VarName284", "VarName285", "VarName286", "VarName287", "VarName288", "VarName289", "VarName290", "VarName291", "VarName292", "VarName293", "VarName294", "VarName295", "VarName296", "VarName297", "VarName298", "VarName299", "VarName300", "VarName301", "VarName302", "VarName303", "VarName304", "VarName305", "VarName306", "VarName307", "VarName308", "VarName309", "VarName310", "VarName311", "VarName312", "VarName313", "VarName314", "VarName315", "VarName316", "VarName317", "VarName318", "VarName319", "VarName320", "VarName321", "VarName322", "VarName323", "VarName324", "VarName325", "VarName326", "VarName327", "VarName328", "VarName329", "VarName330", "VarName331", "VarName332", "VarName333", "VarName334", "VarName335", "VarName336", "VarName337", "VarName338", "VarName339", "VarName340", "VarName341", "VarName342", "VarName343", "VarName344", "VarName345", "VarName346", "VarName347", "VarName348", "VarName349", "VarName350", "VarName351", "VarName352", "VarName353", "VarName354", "VarName355", "VarName356", "VarName357", "VarName358", "VarName359", "VarName360", "VarName361", "VarName362", "VarName363", "VarName364", "VarName365", "VarName366", "VarName367", "VarName368", "VarName369", "VarName370", "VarName371", "VarName372", "VarName373", "VarName374", "VarName375", "VarName376", "VarName377", "VarName378", "VarName379", "VarName380", "VarName381", "VarName382", "VarName383", "VarName384", "VarName385", "VarName386", "VarName387", "VarName388", "VarName389", "VarName390", "VarName391", "VarName392", "VarName393", "VarName394", "VarName395", "VarName396", "VarName397", "VarName398", "VarName399", "VarName400", "VarName401", "VarName402", "VarName403", "VarName404", "VarName405", "VarName406", "VarName407", "VarName408", "VarName409", "VarName410", "VarName411", "VarName412", "VarName413", "VarName414", "VarName415", "VarName416", "VarName417", "VarName418", "VarName419", "VarName420", "VarName421", "VarName422", "VarName423", "VarName424", "VarName425", "VarName426", "VarName427", "VarName428", "VarName429", "VarName430", "VarName431", "VarName432", "VarName433", "VarName434", "VarName435", "VarName436", "VarName437", "VarName438", "VarName439", "VarName440", "VarName441", "VarName442", "VarName443", "VarName444", "VarName445", "VarName446", "VarName447", "VarName448", "VarName449", "VarName450", "VarName451", "VarName452", "VarName453", "VarName454", "VarName455", "VarName456", "VarName457", "VarName458", "VarName459", "VarName460", "VarName461", "VarName462", "VarName463", "VarName464", "VarName465", "VarName466", "VarName467", "VarName468", "VarName469", "VarName470", "VarName471", "VarName472", "VarName473", "VarName474", "VarName475", "VarName476", "VarName477", "VarName478", "VarName479", "VarName480", "VarName481", "VarName482", "VarName483", "VarName484", "VarName485", "VarName486", "VarName487", "VarName488", "VarName489", "VarName490", "VarName491", "VarName492", "VarName493", "VarName494", "VarName495", "VarName496", "VarName497", "VarName498", "VarName499", "VarName500", "VarName501", "VarName502", "VarName503", "VarName504", "VarName505", "VarName506", "VarName507", "VarName508", "VarName509", "VarName510", "VarName511", "VarName512", "VarName513", "VarName514", "VarName515", "VarName516", "VarName517", "VarName518", "VarName519", "VarName520", "VarName521", "VarName522", "VarName523", "VarName524", "VarName525", "VarName526", "VarName527", "VarName528", "VarName529", "VarName530", "VarName531", "VarName532", "VarName533", "VarName534", "VarName535", "VarName536", "VarName537", "VarName538", "VarName539", "VarName540", "VarName541", "VarName542", "VarName543", "VarName544", "VarName545", "VarName546", "VarName547", "VarName548", "VarName549", "VarName550", "VarName551", "VarName552", "VarName553", "VarName554", "VarName555", "VarName556", "VarName557", "VarName558", "VarName559", "VarName560", "VarName561", "VarName562", "VarName563", "VarName564", "VarName565", "VarName566", "VarName567", "VarName568", "VarName569", "VarName570", "VarName571", "VarName572", "VarName573", "VarName574", "VarName575", "VarName576", "VarName577", "VarName578", "VarName579", "VarName580", "VarName581", "VarName582", "VarName583", "VarName584", "VarName585", "VarName586", "VarName587", "VarName588", "VarName589", "VarName590", "VarName591", "VarName592", "VarName593", "VarName594", "VarName595", "VarName596", "VarName597", "VarName598", "VarName599", "VarName600", "VarName601", "VarName602", "VarName603", "VarName604", "VarName605", "VarName606", "VarName607", "VarName608", "VarName609", "VarName610", "VarName611", "VarName612", "VarName613", "VarName614", "VarName615", "VarName616", "VarName617", "VarName618", "VarName619", "VarName620", "VarName621", "VarName622", "VarName623", "VarName624", "VarName625", "VarName626", "VarName627", "VarName628", "VarName629", "VarName630", "VarName631", "VarName632", "VarName633", "VarName634", "VarName635", "VarName636", "VarName637", "VarName638", "VarName639", "VarName640", "VarName641", "VarName642", "VarName643", "VarName644", "VarName645", "VarName646", "VarName647", "VarName648", "VarName649", "VarName650", "VarName651", "VarName652", "VarName653", "VarName654", "VarName655", "VarName656", "VarName657", "VarName658", "VarName659", "VarName660", "VarName661", "VarName662", "VarName663", "VarName664", "VarName665", "VarName666", "VarName667", "VarName668", "VarName669", "VarName670", "VarName671", "VarName672", "VarName673", "VarName674", "VarName675", "VarName676", "VarName677", "VarName678", "VarName679", "VarName680", "VarName681", "VarName682", "VarName683", "VarName684", "VarName685", "VarName686", "VarName687", "VarName688", "VarName689", "VarName690", "VarName691", "VarName692", "VarName693", "VarName694", "VarName695", "VarName696", "VarName697", "VarName698", "VarName699", "VarName700", "VarName701", "VarName702", "VarName703", "VarName704", "VarName705", "VarName706", "VarName707", "VarName708", "VarName709", "VarName710", "VarName711", "VarName712", "VarName713", "VarName714", "VarName715", "VarName716", "VarName717", "VarName718", "VarName719", "VarName720", "VarName721", "VarName722", "VarName723", "VarName724", "VarName725", "VarName726", "VarName727", "VarName728", "VarName729", "VarName730", "VarName731", "VarName732", "VarName733", "VarName734", "VarName735", "VarName736", "VarName737", "VarName738", "VarName739", "VarName740", "VarName741", "VarName742", "VarName743", "VarName744", "VarName745", "VarName746", "VarName747", "VarName748", "VarName749", "VarName750", "VarName751", "VarName752", "VarName753", "VarName754", "VarName755", "VarName756", "VarName757", "VarName758", "VarName759", "VarName760", "VarName761", "VarName762", "VarName763", "VarName764", "VarName765", "VarName766", "VarName767", "VarName768", "VarName769", "VarName770", "VarName771", "VarName772", "VarName773", "VarName774", "VarName775", "VarName776", "VarName777", "VarName778", "VarName779", "VarName780", "VarName781", "VarName782", "VarName783", "VarName784", "VarName785", "VarName786", "VarName787", "VarName788", "VarName789", "VarName790", "VarName791", "VarName792", "VarName793", "VarName794", "VarName795", "VarName796", "VarName797", "VarName798", "VarName799", "VarName800", "VarName801", "VarName802", "VarName803", "VarName804", "VarName805", "VarName806", "VarName807", "VarName808", "VarName809", "VarName810", "VarName811", "VarName812", "VarName813", "VarName814", "VarName815", "VarName816", "VarName817", "VarName818", "VarName819", "VarName820", "VarName821", "VarName822", "VarName823", "VarName824", "VarName825", "VarName826", "VarName827", "VarName828", "VarName829", "VarName830", "VarName831", "VarName832", "VarName833", "VarName834", "VarName835", "VarName836", "VarName837", "VarName838", "VarName839", "VarName840", "VarName841", "VarName842", "VarName843", "VarName844", "VarName845", "VarName846", "VarName847", "VarName848", "VarName849", "VarName850", "VarName851", "VarName852", "VarName853", "VarName854", "VarName855", "VarName856", "VarName857", "VarName858", "VarName859", "VarName860", "VarName861", "VarName862", "VarName863", "VarName864", "VarName865", "VarName866", "VarName867", "VarName868", "VarName869", "VarName870", "VarName871", "VarName872", "VarName873", "VarName874", "VarName875", "VarName876", "VarName877", "VarName878", "VarName879", "VarName880", "VarName881", "VarName882", "VarName883", "VarName884", "VarName885", "VarName886", "VarName887", "VarName888", "VarName889", "VarName890", "VarName891", "VarName892", "VarName893", "VarName894", "VarName895", "VarName896", "VarName897", "VarName898", "VarName899", "VarName900", "VarName901", "VarName902", "VarName903", "VarName904", "VarName905", "VarName906", "VarName907", "VarName908", "VarName909", "VarName910", "VarName911", "VarName912", "VarName913", "VarName914", "VarName915", "VarName916", "VarName917", "VarName918", "VarName919", "VarName920", "VarName921", "VarName922", "VarName923", "VarName924", "VarName925", "VarName926", "VarName927", "VarName928", "VarName929", "VarName930", "VarName931", "VarName932", "VarName933", "VarName934", "VarName935", "VarName936", "VarName937", "VarName938", "VarName939", "VarName940", "VarName941", "VarName942", "VarName943", "VarName944", "VarName945", "VarName946", "VarName947", "VarName948", "VarName949", "VarName950", "VarName951", "VarName952", "VarName953", "VarName954", "VarName955", "VarName956", "VarName957", "VarName958", "VarName959", "VarName960", "VarName961", "VarName962", "VarName963", "VarName964", "VarName965", "VarName966", "VarName967", "VarName968", "VarName969", "VarName970", "VarName971", "VarName972", "VarName973", "VarName974", "VarName975", "VarName976", "VarName977", "VarName978", "VarName979", "VarName980", "VarName981", "VarName982", "VarName983", "VarName984", "VarName985", "VarName986", "VarName987", "VarName988", "VarName989", "VarName990", "VarName991", "VarName992", "VarName993", "VarName994", "VarName995", "VarName996", "VarName997", "VarName998", "VarName999", "VarName1000", "VarName1001"];
opts.SelectedVariableNames = ["VarName2", "VarName3", "VarName4", "VarName5", "VarName6", "VarName7", "VarName8", "VarName9", "VarName10", "VarName11", "VarName12", "VarName13", "VarName14", "VarName15", "VarName16", "VarName17", "VarName18", "VarName19", "VarName20", "VarName21", "VarName22", "VarName23", "VarName24", "VarName25", "VarName26", "VarName27", "VarName28", "VarName29", "VarName30", "VarName31", "VarName32", "VarName33", "VarName34", "VarName35", "VarName36", "VarName37", "VarName38", "VarName39", "VarName40", "VarName41", "VarName42", "VarName43", "VarName44", "VarName45", "VarName46", "VarName47", "VarName48", "VarName49", "VarName50", "VarName51", "VarName52", "VarName53", "VarName54", "VarName55", "VarName56", "VarName57", "VarName58", "VarName59", "VarName60", "VarName61", "VarName62", "VarName63", "VarName64", "VarName65", "VarName66", "VarName67", "VarName68", "VarName69", "VarName70", "VarName71", "VarName72", "VarName73", "VarName74", "VarName75", "VarName76", "VarName77", "VarName78", "VarName79", "VarName80", "VarName81", "VarName82", "VarName83", "VarName84", "VarName85", "VarName86", "VarName87", "VarName88", "VarName89", "VarName90", "VarName91", "VarName92", "VarName93", "VarName94", "VarName95", "VarName96", "VarName97", "VarName98", "VarName99", "VarName100", "VarName101", "VarName102", "VarName103", "VarName104", "VarName105", "VarName106", "VarName107", "VarName108", "VarName109", "VarName110", "VarName111", "VarName112", "VarName113", "VarName114", "VarName115", "VarName116", "VarName117", "VarName118", "VarName119", "VarName120", "VarName121", "VarName122", "VarName123", "VarName124", "VarName125", "VarName126", "VarName127", "VarName128", "VarName129", "VarName130", "VarName131", "VarName132", "VarName133", "VarName134", "VarName135", "VarName136", "VarName137", "VarName138", "VarName139", "VarName140", "VarName141", "VarName142", "VarName143", "VarName144", "VarName145", "VarName146", "VarName147", "VarName148", "VarName149", "VarName150", "VarName151", "VarName152", "VarName153", "VarName154", "VarName155", "VarName156", "VarName157", "VarName158", "VarName159", "VarName160", "VarName161", "VarName162", "VarName163", "VarName164", "VarName165", "VarName166", "VarName167", "VarName168", "VarName169", "VarName170", "VarName171", "VarName172", "VarName173", "VarName174", "VarName175", "VarName176", "VarName177", "VarName178", "VarName179", "VarName180", "VarName181", "VarName182", "VarName183", "VarName184", "VarName185", "VarName186", "VarName187", "VarName188", "VarName189", "VarName190", "VarName191", "VarName192", "VarName193", "VarName194", "VarName195", "VarName196", "VarName197", "VarName198", "VarName199", "VarName200", "VarName201", "VarName202", "VarName203", "VarName204", "VarName205", "VarName206", "VarName207", "VarName208", "VarName209", "VarName210", "VarName211", "VarName212", "VarName213", "VarName214", "VarName215", "VarName216", "VarName217", "VarName218", "VarName219", "VarName220", "VarName221", "VarName222", "VarName223", "VarName224", "VarName225", "VarName226", "VarName227", "VarName228", "VarName229", "VarName230", "VarName231", "VarName232", "VarName233", "VarName234", "VarName235", "VarName236", "VarName237", "VarName238", "VarName239", "VarName240", "VarName241", "VarName242", "VarName243", "VarName244", "VarName245", "VarName246", "VarName247", "VarName248", "VarName249", "VarName250", "VarName251", "VarName252", "VarName253", "VarName254", "VarName255", "VarName256", "VarName257", "VarName258", "VarName259", "VarName260", "VarName261", "VarName262", "VarName263", "VarName264", "VarName265", "VarName266", "VarName267", "VarName268", "VarName269", "VarName270", "VarName271", "VarName272", "VarName273", "VarName274", "VarName275", "VarName276", "VarName277", "VarName278", "VarName279", "VarName280", "VarName281", "VarName282", "VarName283", "VarName284", "VarName285", "VarName286", "VarName287", "VarName288", "VarName289", "VarName290", "VarName291", "VarName292", "VarName293", "VarName294", "VarName295", "VarName296", "VarName297", "VarName298", "VarName299", "VarName300", "VarName301", "VarName302", "VarName303", "VarName304", "VarName305", "VarName306", "VarName307", "VarName308", "VarName309", "VarName310", "VarName311", "VarName312", "VarName313", "VarName314", "VarName315", "VarName316", "VarName317", "VarName318", "VarName319", "VarName320", "VarName321", "VarName322", "VarName323", "VarName324", "VarName325", "VarName326", "VarName327", "VarName328", "VarName329", "VarName330", "VarName331", "VarName332", "VarName333", "VarName334", "VarName335", "VarName336", "VarName337", "VarName338", "VarName339", "VarName340", "VarName341", "VarName342", "VarName343", "VarName344", "VarName345", "VarName346", "VarName347", "VarName348", "VarName349", "VarName350", "VarName351", "VarName352", "VarName353", "VarName354", "VarName355", "VarName356", "VarName357", "VarName358", "VarName359", "VarName360", "VarName361", "VarName362", "VarName363", "VarName364", "VarName365", "VarName366", "VarName367", "VarName368", "VarName369", "VarName370", "VarName371", "VarName372", "VarName373", "VarName374", "VarName375", "VarName376", "VarName377", "VarName378", "VarName379", "VarName380", "VarName381", "VarName382", "VarName383", "VarName384", "VarName385", "VarName386", "VarName387", "VarName388", "VarName389", "VarName390", "VarName391", "VarName392", "VarName393", "VarName394", "VarName395", "VarName396", "VarName397", "VarName398", "VarName399", "VarName400", "VarName401", "VarName402", "VarName403", "VarName404", "VarName405", "VarName406", "VarName407", "VarName408", "VarName409", "VarName410", "VarName411", "VarName412", "VarName413", "VarName414", "VarName415", "VarName416", "VarName417", "VarName418", "VarName419", "VarName420", "VarName421", "VarName422", "VarName423", "VarName424", "VarName425", "VarName426", "VarName427", "VarName428", "VarName429", "VarName430", "VarName431", "VarName432", "VarName433", "VarName434", "VarName435", "VarName436", "VarName437", "VarName438", "VarName439", "VarName440", "VarName441", "VarName442", "VarName443", "VarName444", "VarName445", "VarName446", "VarName447", "VarName448", "VarName449", "VarName450", "VarName451", "VarName452", "VarName453", "VarName454", "VarName455", "VarName456", "VarName457", "VarName458", "VarName459", "VarName460", "VarName461", "VarName462", "VarName463", "VarName464", "VarName465", "VarName466", "VarName467", "VarName468", "VarName469", "VarName470", "VarName471", "VarName472", "VarName473", "VarName474", "VarName475", "VarName476", "VarName477", "VarName478", "VarName479", "VarName480", "VarName481", "VarName482", "VarName483", "VarName484", "VarName485", "VarName486", "VarName487", "VarName488", "VarName489", "VarName490", "VarName491", "VarName492", "VarName493", "VarName494", "VarName495", "VarName496", "VarName497", "VarName498", "VarName499", "VarName500", "VarName501", "VarName502", "VarName503", "VarName504", "VarName505", "VarName506", "VarName507", "VarName508", "VarName509", "VarName510", "VarName511", "VarName512", "VarName513", "VarName514", "VarName515", "VarName516", "VarName517", "VarName518", "VarName519", "VarName520", "VarName521", "VarName522", "VarName523", "VarName524", "VarName525", "VarName526", "VarName527", "VarName528", "VarName529", "VarName530", "VarName531", "VarName532", "VarName533", "VarName534", "VarName535", "VarName536", "VarName537", "VarName538", "VarName539", "VarName540", "VarName541", "VarName542", "VarName543", "VarName544", "VarName545", "VarName546", "VarName547", "VarName548", "VarName549", "VarName550", "VarName551", "VarName552", "VarName553", "VarName554", "VarName555", "VarName556", "VarName557", "VarName558", "VarName559", "VarName560", "VarName561", "VarName562", "VarName563", "VarName564", "VarName565", "VarName566", "VarName567", "VarName568", "VarName569", "VarName570", "VarName571", "VarName572", "VarName573", "VarName574", "VarName575", "VarName576", "VarName577", "VarName578", "VarName579", "VarName580", "VarName581", "VarName582", "VarName583", "VarName584", "VarName585", "VarName586", "VarName587", "VarName588", "VarName589", "VarName590", "VarName591", "VarName592", "VarName593", "VarName594", "VarName595", "VarName596", "VarName597", "VarName598", "VarName599", "VarName600", "VarName601", "VarName602", "VarName603", "VarName604", "VarName605", "VarName606", "VarName607", "VarName608", "VarName609", "VarName610", "VarName611", "VarName612", "VarName613", "VarName614", "VarName615", "VarName616", "VarName617", "VarName618", "VarName619", "VarName620", "VarName621", "VarName622", "VarName623", "VarName624", "VarName625", "VarName626", "VarName627", "VarName628", "VarName629", "VarName630", "VarName631", "VarName632", "VarName633", "VarName634", "VarName635", "VarName636", "VarName637", "VarName638", "VarName639", "VarName640", "VarName641", "VarName642", "VarName643", "VarName644", "VarName645", "VarName646", "VarName647", "VarName648", "VarName649", "VarName650", "VarName651", "VarName652", "VarName653", "VarName654", "VarName655", "VarName656", "VarName657", "VarName658", "VarName659", "VarName660", "VarName661", "VarName662", "VarName663", "VarName664", "VarName665", "VarName666", "VarName667", "VarName668", "VarName669", "VarName670", "VarName671", "VarName672", "VarName673", "VarName674", "VarName675", "VarName676", "VarName677", "VarName678", "VarName679", "VarName680", "VarName681", "VarName682", "VarName683", "VarName684", "VarName685", "VarName686", "VarName687", "VarName688", "VarName689", "VarName690", "VarName691", "VarName692", "VarName693", "VarName694", "VarName695", "VarName696", "VarName697", "VarName698", "VarName699", "VarName700", "VarName701", "VarName702", "VarName703", "VarName704", "VarName705", "VarName706", "VarName707", "VarName708", "VarName709", "VarName710", "VarName711", "VarName712", "VarName713", "VarName714", "VarName715", "VarName716", "VarName717", "VarName718", "VarName719", "VarName720", "VarName721", "VarName722", "VarName723", "VarName724", "VarName725", "VarName726", "VarName727", "VarName728", "VarName729", "VarName730", "VarName731", "VarName732", "VarName733", "VarName734", "VarName735", "VarName736", "VarName737", "VarName738", "VarName739", "VarName740", "VarName741", "VarName742", "VarName743", "VarName744", "VarName745", "VarName746", "VarName747", "VarName748", "VarName749", "VarName750", "VarName751", "VarName752", "VarName753", "VarName754", "VarName755", "VarName756", "VarName757", "VarName758", "VarName759", "VarName760", "VarName761", "VarName762", "VarName763", "VarName764", "VarName765", "VarName766", "VarName767", "VarName768", "VarName769", "VarName770", "VarName771", "VarName772", "VarName773", "VarName774", "VarName775", "VarName776", "VarName777", "VarName778", "VarName779", "VarName780", "VarName781", "VarName782", "VarName783", "VarName784", "VarName785", "VarName786", "VarName787", "VarName788", "VarName789", "VarName790", "VarName791", "VarName792", "VarName793", "VarName794", "VarName795", "VarName796", "VarName797", "VarName798", "VarName799", "VarName800", "VarName801", "VarName802", "VarName803", "VarName804", "VarName805", "VarName806", "VarName807", "VarName808", "VarName809", "VarName810", "VarName811", "VarName812", "VarName813", "VarName814", "VarName815", "VarName816", "VarName817", "VarName818", "VarName819", "VarName820", "VarName821", "VarName822", "VarName823", "VarName824", "VarName825", "VarName826", "VarName827", "VarName828", "VarName829", "VarName830", "VarName831", "VarName832", "VarName833", "VarName834", "VarName835", "VarName836", "VarName837", "VarName838", "VarName839", "VarName840", "VarName841", "VarName842", "VarName843", "VarName844", "VarName845", "VarName846", "VarName847", "VarName848", "VarName849", "VarName850", "VarName851", "VarName852", "VarName853", "VarName854", "VarName855", "VarName856", "VarName857", "VarName858", "VarName859", "VarName860", "VarName861", "VarName862", "VarName863", "VarName864", "VarName865", "VarName866", "VarName867", "VarName868", "VarName869", "VarName870", "VarName871", "VarName872", "VarName873", "VarName874", "VarName875", "VarName876", "VarName877", "VarName878", "VarName879", "VarName880", "VarName881", "VarName882", "VarName883", "VarName884", "VarName885", "VarName886", "VarName887", "VarName888", "VarName889", "VarName890", "VarName891", "VarName892", "VarName893", "VarName894", "VarName895", "VarName896", "VarName897", "VarName898", "VarName899", "VarName900", "VarName901", "VarName902", "VarName903", "VarName904", "VarName905", "VarName906", "VarName907", "VarName908", "VarName909", "VarName910", "VarName911", "VarName912", "VarName913", "VarName914", "VarName915", "VarName916", "VarName917", "VarName918", "VarName919", "VarName920", "VarName921", "VarName922", "VarName923", "VarName924", "VarName925", "VarName926", "VarName927", "VarName928", "VarName929", "VarName930", "VarName931", "VarName932", "VarName933", "VarName934", "VarName935", "VarName936", "VarName937", "VarName938", "VarName939", "VarName940", "VarName941", "VarName942", "VarName943", "VarName944", "VarName945", "VarName946", "VarName947", "VarName948", "VarName949", "VarName950", "VarName951", "VarName952", "VarName953", "VarName954", "VarName955", "VarName956", "VarName957", "VarName958", "VarName959", "VarName960", "VarName961", "VarName962", "VarName963", "VarName964", "VarName965", "VarName966", "VarName967", "VarName968", "VarName969", "VarName970", "VarName971", "VarName972", "VarName973", "VarName974", "VarName975", "VarName976", "VarName977", "VarName978", "VarName979", "VarName980", "VarName981", "VarName982", "VarName983", "VarName984", "VarName985", "VarName986", "VarName987", "VarName988", "VarName989", "VarName990", "VarName991", "VarName992", "VarName993", "VarName994", "VarName995", "VarName996", "VarName997", "VarName998", "VarName999", "VarName1000", "VarName1001"];
opts.VariableTypes = ["string", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, "Var1", "WhitespaceRule", "preserve");
opts = setvaropts(opts, "Var1", "EmptyFieldRule", "auto");

% Import the data
raman_spec_valid = readtable("C:\Users\Rob Roy\OneDrive - Maynooth University\Documents\MATLAB\4dsigma1new_0_24std_Raman_spectrums_valid.csv", opts);


%% Clear temporary variables
clear opts