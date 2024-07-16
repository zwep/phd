classdef ProcessLabFile
    properties
        file_name
        recon_obj
        sense_file
        dest_file
    end
    
    methods
        function obj = ProcessLabFile(file_name, sense_file, dest_file)
            obj.file_name = file_name;
            obj.dest_file = dest_file;
            obj.sense_file = sense_file;
            obj.recon_obj = MRecon(file_name);
        end
        
        function recon_data = ProcessData(obj)
            recon_data = NaN;
            check_data_typ = any(ismember(obj.recon_obj.Parameter.Parameter2Read.typ, 1));
            if check_data_typ
                if isempty(obj.recon_obj.Parameter.Scan.ScanType)
                    fprintf('\t No Recon parameters found\n')
                else
                    sprintf('\t Recon parameters found')
                    if strcmp(obj.sense_file, 'none')
                        fprintf('\t No REF file yet.\n')
                        obj.recon_obj.Perform   
                        disp('Done - calculated Perform unit')
                    else
                        fprintf('\t REF file found.\n')
                        S = MRsense(obj.sense_file, path_file);
                        % Calculate the body coil from this data
                        body_abs = sum(abs(S.CoilData), 4);
                        body_angle = angle(sum(S.CoilData, 4));
                        body_coil_combined = body_abs .* exp(1i * body_angle);
                        S.BodycoilData = body_coil_combined ;  
                        S.Perform
                        disp('Done - calculated MR Sense unit')

                        obj.recon_obj.Parameter.Recon.Sensitivities = S;
                        obj.recon_obj.Perform
                        disp('Done - calculated Perform unit')
                        toc
                    end
                    recon_data = obj.recon_obj.Data;
                    
                end      
            else
                fprintf('No data typ 1 available\n')
            end
        end
        
        function SaveData(obj, data)
            save(obj.dest_file, data);
            disp('Done - saving reconstruction data')
        end
        
    end
end
