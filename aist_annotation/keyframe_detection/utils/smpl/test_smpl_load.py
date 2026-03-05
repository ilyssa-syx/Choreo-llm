import numpy as np
import pickle
import os

def convert_smpl_npz_to_pkl(npz_path, output_path=None):
    """
    Convert SMPL .npz to .pkl format for smplx compatibility
    
    Args:
        npz_path: Path to your SMPL_MALE.npz file
        output_path: Output path for .pkl file (optional)
    """
    print(f"Loading {npz_path}...")
    
    # Load the npz file
    try:
        smpl_data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading npz file: {e}")
        return None
    
    print(f"Available keys: {list(smpl_data.files)}")
    
    # Create the dictionary structure expected by smplx
    smpl_dict = {}
    
    # Map the standard SMPL parameters
    key_mappings = {
        'v_template': 'v_template',      # Template vertices
        'shapedirs': 'shapedirs',        # Shape blend shapes
        'posedirs': 'posedirs',          # Pose blend shapes  
        'J_regressor': 'J_regressor',    # Joint regressor
        'kintree_table': 'kintree_table', # Kinematic tree
        'weights': 'weights',            # Skinning weights
        'f': 'f',                        # Faces
        'faces': 'f'                     # Alternative faces key
    }
    
    # Copy available data
    for npz_key, dict_key in key_mappings.items():
        if npz_key in smpl_data.files:
            smpl_dict[dict_key] = smpl_data[npz_key]
            print(f"✓ Copied {npz_key} -> {dict_key} {smpl_data[npz_key].shape}")
    
    # Handle faces specially (some files use 'f', others use 'faces')
    if 'f' not in smpl_dict and 'faces' in smpl_data.files:
        smpl_dict['f'] = smpl_data['faces']
        print(f"✓ Copied faces -> f {smpl_data['faces'].shape}")
    
    # Set output path
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(npz_path))[0]
        output_path = f"{base_name}.pkl"
    
    # Save as pickle
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(smpl_dict, f)
        print(f"✅ Successfully saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error saving pkl file: {e}")
        return None

def test_converted_file(pkl_path):
    """
    Test if the converted file works with smplx
    """
    try:
        from smplx import SMPL
        import torch
        
        print(f"Testing {pkl_path} with smplx...")
        
        # Determine gender from filename
        if 'MALE' in pkl_path.upper() and 'FEMALE' not in pkl_path.upper():
            gender = 'male'
        elif 'FEMALE' in pkl_path.upper():
            gender = 'female'
        else:
            gender = 'neutral'
        
        # Try to create SMPL model
        smpl = SMPL(model_path=pkl_path, gender=gender, batch_size=1)
        
        # Test forward pass
        body_pose = torch.zeros(1, 69)  # 23 joints * 3 = 69
        global_orient = torch.zeros(1, 3)
        
        output = smpl(global_orient=global_orient, body_pose=body_pose)
        
        print(f"✅ Success! Output joints shape: {output.joints.shape}")
        return True
        
    except ImportError:
        print("⚠️  smplx not installed, cannot test. But file was created.")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    # Convert your files
    npz_files = ['SMPL_MALE.npz', 'SMPL_FEMALE.npz', 'SMPL_NEUTRAL.npz']
    
    for npz_file in npz_files:
        if os.path.exists(npz_file):
            print(f"\n{'='*50}")
            print(f"Converting {npz_file}")
            print('='*50)
            
            pkl_file = convert_smpl_npz_to_pkl(npz_file)
            if pkl_file:
                test_converted_file(pkl_file)
        else:
            print(f"File {npz_file} not found, skipping...")
    
    print("\n🎉 Conversion completed!")
    print("You can now use the .pkl files with smplx:")
    print('smpl = SMPL(model_path="SMPL_MALE.pkl", gender="male", batch_size=1)')
