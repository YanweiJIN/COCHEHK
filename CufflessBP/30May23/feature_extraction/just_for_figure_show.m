clc;
clear all;
close all;

% setting
disp_original_BP_ECG_PPG=0; % 0 not display, 1 display
disp_BP_PPG_ECG_annotation=1; % 0 not display, 1 display
movemedian_window=10; % number of elements considered for outlier detection and missing data filling
tolerance=0.2; % for determination of number of misssing timestep
shift_beats=32; % the number of beats of shift of two ajacent sample, when shift_beats==sample_beats, samples are unoverlapped
sample_beats=32; % the number of beats in one sample
Rpeak_half_interval=5; % the length of half adjust interval of Rpeak (under 200 Hz setting)

directory='/media/darcy/Documents/code/shenzhen_task_1';
subdir_storedata='/new_feature_data_Jan_14th_PTT2BP';
name_cell0={'shenzixiao','funan_rest','funan_sport','hongxi_day_1','hongxi_day_2','hongxi_day_3','hongxi_day_4'};

mkdir(strcat(directory,subdir_storedata));
for k=[1:1:size(name_cell0,2)]
    mkdir(strcat(directory,subdir_storedata,'/',name_cell0{1,k}));
end

for(p=[1:1:7])  % source
    if(p==1)
        q_max=62;
    elseif(p==2 | p==3)
        q_max=45;
    elseif(p>=4 & p<=7)
        q_max=12;
    end
    for(q=1:1:q_max) % person 
        % setting
       disp(strcat('p=',num2str(p),'q=',num2str(q)));
        % load data
        S=load(strcat(directory,'/personal_conditional_data_mat','/',name_cell0{1,p},'/','person_',num2str(q),'_',name_cell0{1,p},'_data.mat'));
        data=S.data;
        BP=data(:,1);
        ECG=data(:,2);
        PPG=data(:,3);
        % filter BP, ECG, PPG
        fs=200; % here original frequency is 200 Hz
        BP=lowpass(BP,30,fs);
        ECG=lowpass(ECG,30,fs);
        PPG=lowpass(PPG,30, fs);
        % display original BP, ECG and PPG
        summary=horzcat(BP,ECG,PPG);
        len_summary=size(summary,1);
        name_cell1={'BP','ECG','PPG'};
        if(disp_original_BP_ECG_PPG)
            figure;
            for k=[1:1:3]
                ax(k)=subplot(3,1,k);
                plot([1:1:len_summary],summary(:,k));
                ylabel(name_cell1{k});
                xlabel('time')
            end
            linkaxes(ax,'x');
        end
        
        BP=BP.';
        ECG=ECG.';
        PPG=PPG.';
        summary=summary.';
        % find the foots and peaks of BP, ECG and PPG
        % here all are 200Hz
        % the detection of ECG and BP seems to be robust, while the
        % detection of PPG is not good due to the variation of PPG shape
        disp('Detecting ECG...');
        [qrs_amp_raw,RpeakIndex_ECG,delay]=pan_tompkin(ECG,200,0);
        disp('Done.');
        [ footIndex_BP, systolicIndex_BP, notchIndex_BP, dicroticIndex_BP ] = ...
         BP_annotate( BP, 200, 0, 'mmHg', 1);
     
        % small adjustment of Rpeaks location due to filter phase shift
        temp_RpeakIndex_ECG=[];
        for k=[1:1:size(RpeakIndex_ECG,2)]
           temp_local_peak=[];
           temp_local_peak_index=[];
           adjust_interval=[max(1, RpeakIndex_ECG(k)-Rpeak_half_interval):min(size(ECG,2),RpeakIndex_ECG(k)+Rpeak_half_interval)];
           [temp_local_peak, temp_local_peak_index]=findpeaks(ECG(adjust_interval));
           [temp_max_peak, temp_max_peak_index]=max(temp_local_peak);
           temp_RpeakIndex_ECG=[temp_RpeakIndex_ECG, adjust_interval(temp_local_peak_index(temp_max_peak_index)) ];
        end
        RpeakIndex_ECG=temp_RpeakIndex_ECG;
%         figure;
%         plot(ECG);
%         hold on;
%         plot(temp_RpeakIndex_ECG, ECG(temp_RpeakIndex_ECG));
        
        % detection of foot and peak of PPG
        temp_peakIndex_PPG=[];
        for k=[1:1:size(RpeakIndex_ECG,2)-1]
           temp_local_peak=[];
           temp_local_peak_index=[];
           beat_interval=[RpeakIndex_ECG(k):RpeakIndex_ECG(k+1)];
           [temp_local_peak, temp_local_peak_index]=findpeaks(PPG(beat_interval));
           [temp_max_peak, temp_max_peak_index]=max(temp_local_peak);
           temp_peakIndex_PPG=[temp_peakIndex_PPG, beat_interval(temp_local_peak_index(temp_max_peak_index)) ];
        end
        peakIndex_PPG=temp_peakIndex_PPG;
%         figure;
%         plot(PPG);
%         hold on;
%         plot(peakIndex_PPG, PPG(peakIndex_PPG));
%         hold on;
        
        % delete repeat value
        time_markers={RpeakIndex_ECG, footIndex_BP, systolicIndex_BP, ...
            notchIndex_BP, dicroticIndex_BP, peakIndex_PPG};
        for (k=[1:1:6])
            time_markers{1,k}=unique(time_markers{1,k});
        end
       
        % detect outlier, determine missing timestep, fill missing data 
        % total_index_data={RpeakIndexname_cell_ECG,footIndex_BP,footIndex_PPG,systolicIndex_BP,systolicIndex_PPG,notchIndex_BP,notchIndex_PPG,dicroticIndex_BP,dicroticIndex_PPG};
        name_cell2={'RpeakIndex-ECG','footIndex-BP','systolicIndex-BP','notchIndex-BP','dicroticIndex-BP','peakIndex-PPG'};
        type={2,1,1,1,1,3 }; %1:BP, 2:ECG, 3:PPG   
        for(i=[1:1:6])
            index_data_name=name_cell2{i};  
            original_data=summary(:,type{i});
            index_data=time_markers{i};
            % detect outlier
            index_data_diff=diff(index_data(1:end));
            % determine missing timestep and fill missing index_data
            outlier=find(isoutlier(index_data_diff,'movmedian',movemedian_window));
            ref_period=movmedian(index_data_diff,movemedian_window);
            % % note that the missing time step is between index_data(outlier(k)) and index_data(outlier(k)+1)
            insert_all_index=[];
            for k=[1:1:size(outlier,2)]
                current_ref_period=ref_period(outlier(k));
                thispoint=index_data(outlier(k)); % the last point before missing index_data
                nextpoint=index_data(outlier(k)+1); % the first point after missing index_data
                gap=nextpoint-thispoint;
                insert_flag=1;
                insert_index=[];
                if(current_ref_period*(2-tolerance)>gap)
                    insert_flag=0; % the gap is too small, no need to insert
                end   
                if(insert_flag)
                    n=2;  % n represent the number of current_ref_period in gap roughly
                    while(1)
                        if( (current_ref_period*(n-tolerance)<=gap) & (current_ref_period*(n+tolerance)>=gap) )
                            break;
                        elseif(current_ref_period*(n+1-tolerance)>gap)
                            break;
                        end
                        n=n+1;
                    end            
                    for r=[1:1:n-1] % the number of inset point should be n-1
                        % note that we should fill index rather than orignal data
                        insert_index=[insert_index, round(thispoint+r*current_ref_period)];
                    end
                end
                insert_all_index=[insert_all_index, insert_index];      
            end
            % insert_all_indexs is used for following data filling
            insert_all_indexs{i}=insert_all_index;
            index_data=sort([index_data,insert_all_index]);
            index_data_temp=unique(index_data);
            assert(isequal(index_data_temp,index_data));
            time_markers{i}=index_data;
        end
        RpeakIndex_ECG=time_markers{1};
        footIndex_BP=time_markers{2};       
        systolicIndex_BP=time_markers{3};    
        notchIndex_BP=time_markers{4};    
        dicroticIndex_BP=time_markers{5};
        peakIndex_PPG=time_markers{6};
        

        % eliminate the incomplete first and last period
        % eliminate the incomplete first period
        % % set the ECG R peak to be the reference
        % % Each pulse should be ECG R peak, PPG(BP) foot, PPG(BP) systolic peak, PPG(BP) notch, PPG(BP) dicrotic peak in order
        RpeakIndex_ECG=RpeakIndex_ECG(2:end);
        while(peakIndex_PPG(1)<RpeakIndex_ECG(1))
            peakIndex_PPG=peakIndex_PPG(2:end);
        end
        % % % note that the comparison standard is little different
        [temp,posi]=min(abs(footIndex_BP-RpeakIndex_ECG(1)));
        footIndex_BP=footIndex_BP(posi:end);
        while(systolicIndex_BP(1)<footIndex_BP(1))
            systolicIndex_BP=systolicIndex_BP(2:end);
        end
        while(notchIndex_BP(1)<systolicIndex_BP(1))
            notchIndex_BP=notchIndex_BP(2:end);
        end
        while(dicroticIndex_BP(1)<notchIndex_BP(1))
            dicroticIndex_BP=dicroticIndex_BP(2:end);
        end

%         % eliminate the redundant points in each period
        temp1=[RpeakIndex_ECG(1)];
        temp2=[];
        temp3=[];
        temp4=[];
        temp5=[];
        temp6=[];
        k=1;
        while(1)
            temp0=peakIndex_PPG(peakIndex_PPG>temp1(k));
            if(size(temp0,2)==0)
                break;
            else
                temp6=[temp6,temp0(1)];
            end
            temp0=RpeakIndex_ECG(RpeakIndex_ECG>temp6(k));
            if(size(temp0,2)==0)
                break;
            else
                temp1=[temp1,temp0(1)];
            end
            k=k+1;
        end
        for(k=[1:1:size(temp1,2)]) % a little different, here we don't change temp1
           [temp,posi]=min(abs(footIndex_BP-temp1(k)));
            temp2=[temp2,footIndex_BP(posi)];
            temp0=systolicIndex_BP(systolicIndex_BP>temp2(k));
            if(size(temp0,2)==0)
                break;
            else
                temp3=[temp3,temp0(1)];
            end
            temp0=notchIndex_BP(notchIndex_BP>temp3(1,k));
            if(size(temp0,2)==0)
                break;
            else
                temp4=[temp4,temp0(1,1)];
            end
            temp0=dicroticIndex_BP(dicroticIndex_BP>temp4(1,k));
            if(size(temp0,2)==0)
                break;
            else
                temp5=[temp5,temp0(1,1)];
            end
        end
        
        % delete the last incomplete periofd
        len_index=min([size(temp1,2),size(temp2,2),size(temp3,2),size(temp4,2),size(temp5,2),size(temp6,2)]);
        RpeakIndex_ECG=temp1(1:len_index);
        footIndex_BP=temp2(1:1:len_index);
        systolicIndex_BP=temp3(1:1:len_index);
        notchIndex_BP=temp4(1:1:len_index);
        dicroticIndex_BP=temp5(1:1:len_index);
        peakIndex_PPG=temp6(1:1:len_index);
        
        % calculate number of spoints of each signal, display
        statistic=[size(RpeakIndex_ECG,2),size(footIndex_BP,2),size(systolicIndex_BP,2),...
            size(notchIndex_BP,2), size(dicroticIndex_BP,2), size(peakIndex_PPG,2)];
        if( size(find(statistic(statistic>=max(statistic))),2) <size(statistic,2))
            disp('data point not in same number!!!!');
            disp(p);
            disp(q);
        end
        
        % here the SBP, DBP and MBP contain all the SBP, DBP and MBP
        % get original SBP, DBP point and fill the abnormal with current movemdian value
        % original SBP, DBP
        SBP_origin=[];
        DBP_origin=[];
        SBP=[];
        DBP=[];
        MBP=[];
        for (k=[1:1:max(statistic)])
            % here the SBP, DBP and MBP contain all the SBP, DBP and MBP
            SBP_origin(k)=BP(systolicIndex_BP(k));
            DBP_origin(k)=BP(footIndex_BP(k));
        end
        SBP_movemedian=movmedian(SBP_origin, movemedian_window);
        DBP_movemedian=movmedian(DBP_origin, movemedian_window);
        for (k=[1:1:max(statistic)])
            %name_cell2={'RpeakIndex-ECG','footIndex-BP','footIndex-PPG','systolicIndex-BP','systolicIndex-PPG','notchIndex-BP','notchIndex-PPG','dicroticIndex-BP','dicroticIndex-PPG'};
            if ismember(systolicIndex_BP(k),insert_all_indexs{3})
                SBP(k)=SBP_movemedian(k);
            else
                SBP(k)=SBP_origin(k);
            end
            if ismember(footIndex_BP(k), insert_all_indexs{2})
                DBP(k)=DBP_movemedian(k);
            else
                DBP(k)=DBP_origin(k);
            end
            MBP(k)=2/3*DBP(k)+1/3*SBP(k);
        end
        
        
        
        %check the timemarker location
        if disp_BP_PPG_ECG_annotation==1
            figure;
            sub(1)=subplot(3,1,1);
            plot([1:1:size(BP,2)],BP);
            hold on;
            plot(footIndex_BP([1:2:size(footIndex_BP,2)]),BP(footIndex_BP([1:2:size(footIndex_BP,2)])), 'o','color','r');
            hold on;
            plot(footIndex_BP([2:2:size(footIndex_BP,2)]),BP(footIndex_BP([2:2:size(footIndex_BP,2)])), 'o','color','g');
            hold on;
            plot(dicroticIndex_BP([2:2:size(dicroticIndex_BP,2)]),BP(dicroticIndex_BP([2:2:size(dicroticIndex_BP,2)])), 'o','color','g');
            ylabel('BP');
            xlabel('time');
            plot(systolicIndex_BP([1:2:size(systolicIndex_BP,2)]),BP(systolicIndex_BP([1:2:size(systolicIndex_BP,2)])), 'o','color','r');
            hold on;
            plot(systolicIndex_BP([2:2:size(systolicIndex_BP,2)]),BP(systolicIndex_BP([2:2:size(systolicIndex_BP,2)])), 'o','color','g');
            hold on;
            plot(notchIndex_BP([1:2:size(notchIndex_BP,2)]),BP(notchIndex_BP([1:2:size(notchIndex_BP,2)])), 'o','color','r');
            hold on;
            plot(notchIndex_BP([2:2:size(notchIndex_BP,2)]),BP(notchIndex_BP([2:2:size(notchIndex_BP,2)])), 'o','color','g');
            hold on;
            plot(dicroticIndex_BP([1:2:size(dicroticIndex_BP,2)]),BP(dicroticIndex_BP([1:2:size(dicroticIndex_BP,2)])), 'o','color','r');
            hold on;
            plot(dicroticIndex_BP([1:2:size(dicroticIndex_BP,2)]),BP(dicroticIndex_BP([1:2:size(dicroticIndex_BP,2)])), 'o','color','r');
            hold on;
            sub(2)=subplot(3,1,2);
            plot([1:1:size(ECG,2)],ECG);
            hold on;
            plot(RpeakIndex_ECG([1:2:size(RpeakIndex_ECG,2)]),ECG(RpeakIndex_ECG([1:2:size(RpeakIndex_ECG,2)])), 'o','color','r');
            hold on;
            plot(RpeakIndex_ECG([2:2:size(RpeakIndex_ECG,2)]),ECG(RpeakIndex_ECG([2:2:size(RpeakIndex_ECG,2)])), 'o','color','g');
            ylabel('ECG');
            xlabel('time');
            sub(3)=subplot(3,1,3);
            plot([1:1:size(PPG,2)],PPG);
            hold on;
            plot(peakIndex_PPG([2:2:size(peakIndex_PPG,2)]),PPG(peakIndex_PPG([2:2:size(peakIndex_PPG,2)])), 'o','color','g');
            hold on;
            plot(peakIndex_PPG([1:2:size(peakIndex_PPG,2)]),PPG(peakIndex_PPG([1:2:size(peakIndex_PPG,2)])), 'o','color','r');
            hold on;
            ylabel('PPG');
            xlabel('time');
            linkaxes(sub,'x');
        end
        
        PTT=[];
        PTT_normalized=[];
        for k=[1:1:max(statistic)]
            PTT(k)=peakIndex_PPG(k)-RpeakIndex_ECG(k);
        end
        PTT=filloutliers(PTT,'linear','movmedian',movemedian_window);
        PTT_normalized=normalize(PTT);
        feature_person=vertcat(PTT_normalized, SBP, DBP, MBP);        
        feature_person=feature_person.';
        sample_num=floor((size(feature_person,1)-sample_beats)/shift_beats); % here one sample has 64 groups of features
        sample_order=0;
        for(k=[1:1:sample_num])        
            feature=feature_person(1+(k-1)*shift_beats:(k-1)*shift_beats+sample_beats,:);
            sample_order=sample_order+1;
            feature_mat=strcat(directory,subdir_storedata,'/',name_cell0{1,p},'/person_',num2str(q),'_sample_',num2str(sample_order),'_',name_cell0{1,p},'_feature.mat');
            save(feature_mat,'feature');
        end
    end
end