classdef SebKspaceReconNoSort < MRecon   % Derive My_Recon from MRecon    
    properties
        % No additional properties needed
    end
    
    methods
        function MR = Seb_Kspace_Recon_no_sort( filename )
            % Create an MRecon object MR upon creation of an My_Recon
            % object
            MR = MR@MRecon(filename); 
        end
        
        % Overload (overwrite) the existing Perform function of MRecon    
        function Perform( MR )            
            %Reconstruct only standard (imaging) data
            MR.Parameter.Parameter2Read.typ = 1;
            MR.Parameter.Parameter2Read.Update;
            MR.ReadData;
            MR.RandomPhaseCorrection;
            MR.RemoveOversampling;
            MR.PDACorrection;
            MR.DcOffsetCorrection;
            MR.MeasPhaseCorrection;
        end
        
    end
end


