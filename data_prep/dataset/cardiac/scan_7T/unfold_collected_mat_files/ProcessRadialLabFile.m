classdef ProcessRadialLabFile < handle
    % With this we can read a .lab file (specifically radial)
    properties
        file_name
        recon_obj
        dest_dir
        dest_name
        proper_read
    end
    
    methods
        function obj = ProcessRadialLabFile(source_file, dest_dir, dest_name)
            obj.file_name = source_file;
            obj.dest_dir = dest_dir;
            obj.dest_name = dest_name;
            obj.recon_obj = MRecon(source_file);
            obj.proper_read = false;
            % Create the destination directory
            mkdir(obj.dest_dir)
        end
        
        function ProcessData(obj)
            try
                obj.recon_obj.ReadData;
                obj.recon_obj.RandomPhaseCorrection;
                obj.recon_obj.RemoveOversampling;
                obj.recon_obj.PDACorrection;
                obj.recon_obj.DcOffsetCorrection;
                obj.recon_obj.MeasPhaseCorrection;
                obj.proper_read = true;
                fprintf(1,'Everything is fine');
            catch e
                fprintf(1,'The identifier was:\n%s',e.identifier);
                fprintf(1,'There was an error! The message was:\n%s',e.message);
                obj.proper_read = false;
            end
        end
        
        function SaveData(obj)
            fprintf(1, 'The object was read: %s\n\n', mat2str(obj.proper_read));
            if obj.proper_read
                unsorted_data = obj.recon_obj.Data;
                data_name = strcat(obj.dest_name, '_data.mat');
                label_name = strcat(obj.dest_name, '_label.list');
                save(fullfile(obj.dest_dir, data_name), 'unsorted_data');
                obj.recon_obj.ExportLabels(fullfile(obj.dest_dir, label_name))
                disp('Done - saving reconstruction data')                
            else
                disp('Woops.. Reading went wrong. On to the next!')
                
            end
        end
        
    end
end
