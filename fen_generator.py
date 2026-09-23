"""
FEN Notation Generation for Chess Recognition
Based on "Determining Chess Game State From an Image" by Wölflin & Arandjelović

This module converts the 8x8 chess board state into standard FEN notation.
FEN (Forsyth-Edwards Notation) is the standard notation for describing chess positions.
"""

import numpy as np
from typing import List, Optional, Tuple

# Mapping from piece class names to FEN characters
PIECE_TO_FEN = {
    'white_pawn': 'P',
    'white_knight': 'N', 
    'white_bishop': 'B',
    'white_rook': 'R',
    'white_queen': 'Q',
    'white_king': 'K',
    'black_pawn': 'p',
    'black_knight': 'n',
    'black_bishop': 'b', 
    'black_rook': 'r',
    'black_queen': 'q',
    'black_king': 'k'
}

# Reverse mapping for FEN to piece names
FEN_TO_PIECE = {v: k for k, v in PIECE_TO_FEN.items()}

class FENGenerator:
    """Class for generating FEN notation from chess board state"""
    
    def __init__(self):
        self.piece_to_fen = PIECE_TO_FEN
        self.fen_to_piece = FEN_TO_PIECE
    
    def board_to_fen(self, board_state: np.ndarray, 
                    active_color: str = 'w',
                    castling_rights: str = '-',
                    en_passant: str = '-',
                    halfmove_clock: int = 0,
                    fullmove_number: int = 1) -> str:
        """
        Convert 8x8 board state to complete FEN notation
        
        Args:
            board_state: 8x8 numpy array with piece names or FEN characters
            active_color: 'w' for white, 'b' for black
            castling_rights: Castling availability (KQkq or combinations)
            en_passant: En passant target square in algebraic notation
            halfmove_clock: Number of halfmoves since last capture or pawn advance
            fullmove_number: Number of the full move
            
        Returns:
            Complete FEN string
        """
        # Convert board state to piece placement string
        piece_placement = self._board_to_piece_placement(board_state)
        
        # Construct complete FEN string
        fen_parts = [
            piece_placement,
            active_color,
            castling_rights,
            en_passant,
            str(halfmove_clock),
            str(fullmove_number)
        ]
        
        return ' '.join(fen_parts)
    
    def _board_to_piece_placement(self, board_state: np.ndarray) -> str:
        """
        Convert 8x8 board state to FEN piece placement string
        
        Args:
            board_state: 8x8 array where each cell contains:
                         - piece name (e.g., 'white_pawn', 'black_king')
                         - FEN character (e.g., 'P', 'k')
                         - empty indicator (None, '', 0, or 'empty')
        
        Returns:
            FEN piece placement string (first part of FEN notation)
        """
        fen_rows = []
        
        # Process each rank (row) from 8th rank (index 0) to 1st rank (index 7)
        for row in range(8):
            fen_row = ""
            empty_count = 0
            
            # Process each file (column) from a-file (index 0) to h-file (index 7)
            for col in range(8):
                cell = board_state[row, col]
                fen_char = self._cell_to_fen_character(cell)
                
                if fen_char is None:  # Empty square
                    empty_count += 1
                else:  # Occupied square
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += fen_char
            
            # Add any remaining empty squares at the end of the row
            if empty_count > 0:
                fen_row += str(empty_count)
            
            fen_rows.append(fen_row)
        
        # Join rows with '/' separator
        return '/'.join(fen_rows)
    
    def _cell_to_fen_character(self, cell) -> Optional[str]:
        """
        Convert a single cell value to FEN character
        
        Args:
            cell: Cell value (piece name, FEN character, or empty indicator)
            
        Returns:
            FEN character or None for empty square
        """
        if cell is None or cell == '' or cell == 0 or cell == 'empty':
            return None
        
        # If already a FEN character
        if isinstance(cell, str) and len(cell) == 1 and cell in self.fen_to_piece:
            return cell
        
        # If it's a piece name
        if isinstance(cell, str) and cell in self.piece_to_fen:
            return self.piece_to_fen[cell]
        
        # If it's a numeric class index (0-11 for 12 piece classes)
        if isinstance(cell, (int, np.integer)) and 0 <= cell <= 11:
            piece_names = list(self.piece_to_fen.keys())
            if cell < len(piece_names):
                return self.piece_to_fen[piece_names[cell]]
        
        # Default to empty if unknown
        return None
    
    def fen_to_board(self, fen: str) -> Tuple[np.ndarray, dict]:
        """
        Convert FEN notation to 8x8 board state array
        
        Args:
            fen: Complete FEN string
            
        Returns:
            Tuple of (board_state, fen_info) where:
            - board_state: 8x8 numpy array with piece names
            - fen_info: Dictionary with FEN components
        """
        fen_parts = fen.strip().split()
        
        if len(fen_parts) != 6:
            raise ValueError(f"Invalid FEN notation: {fen}")
        
        piece_placement = fen_parts[0]
        
        # Parse FEN components
        fen_info = {
            'piece_placement': piece_placement,
            'active_color': fen_parts[1],
            'castling_rights': fen_parts[2],
            'en_passant': fen_parts[3],
            'halfmove_clock': int(fen_parts[4]),
            'fullmove_number': int(fen_parts[5])
        }
        
        # Convert piece placement to board array
        board_state = self._piece_placement_to_board(piece_placement)
        
        return board_state, fen_info
    
    def _piece_placement_to_board(self, piece_placement: str) -> np.ndarray:
        """Convert FEN piece placement string to 8x8 board array"""
        board = np.full((8, 8), 'empty', dtype=object)
        
        ranks = piece_placement.split('/')
        if len(ranks) != 8:
            raise ValueError(f"Invalid piece placement: {piece_placement}")
        
        for rank_idx, rank in enumerate(ranks):
            file_idx = 0
            
            for char in rank:
                if char.isdigit():
                    # Empty squares
                    empty_count = int(char)
                    for _ in range(empty_count):
                        if file_idx < 8:
                            board[rank_idx, file_idx] = 'empty'
                            file_idx += 1
                else:
                    # Piece
                    if file_idx < 8 and char in self.fen_to_piece:
                        board[rank_idx, file_idx] = self.fen_to_piece[char]
                        file_idx += 1
        
        return board
    
    def validate_fen(self, fen: str) -> Tuple[bool, List[str]]:
        """
        Validate FEN notation and return any errors
        
        Args:
            fen: FEN string to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            board_state, fen_info = self.fen_to_board(fen)
            
            # Check piece counts
            piece_counts = self._count_pieces(board_state)
            
            # Validate piece counts according to chess rules
            validation_errors = self._validate_piece_counts(piece_counts)
            errors.extend(validation_errors)
            
            # Check king presence
            if piece_counts.get('white_king', 0) != 1:
                errors.append("Must have exactly one white king")
            if piece_counts.get('black_king', 0) != 1:
                errors.append("Must have exactly one black king")
            
            # Check pawn positions
            pawn_errors = self._validate_pawn_positions(board_state)
            errors.extend(pawn_errors)
            
        except Exception as e:
            errors.append(f"FEN parsing error: {str(e)}")
        
        return len(errors) == 0, errors
    
    def _count_pieces(self, board_state: np.ndarray) -> dict:
        """Count pieces on the board"""
        piece_counts = {}
        
        for row in range(8):
            for col in range(8):
                piece = board_state[row, col]
                if piece != 'empty' and piece is not None:
                    piece_counts[piece] = piece_counts.get(piece, 0) + 1
        
        return piece_counts
    
    def _validate_piece_counts(self, piece_counts: dict) -> List[str]:
        """Validate piece counts according to chess rules"""
        errors = []
        
        # Maximum piece counts
        max_counts = {
            'white_pawn': 8, 'black_pawn': 8,
            'white_knight': 2, 'black_knight': 2,
            'white_bishop': 2, 'black_bishop': 2,
            'white_rook': 2, 'black_rook': 2,
            'white_queen': 1, 'black_queen': 1,  # Can be more due to promotion
            'white_king': 1, 'black_king': 1
        }
        
        # Check maximum counts (allowing for promotion)
        for piece, max_count in max_counts.items():
            if piece in ['white_queen', 'black_queen']:
                continue  # Queens can be promoted, so skip this check
            
            actual_count = piece_counts.get(piece, 0)
            if actual_count > max_count:
                errors.append(f"Too many {piece}: {actual_count} (max {max_count})")
        
        return errors
    
    def _validate_pawn_positions(self, board_state: np.ndarray) -> List[str]:
        """Validate pawn positions (not on first or last rank)"""
        errors = []
        
        # Check first rank (rank 8, index 0)
        for col in range(8):
            piece = board_state[0, col]
            if piece in ['white_pawn', 'black_pawn']:
                errors.append(f"Pawn on rank 8 at file {chr(ord('a') + col)}")
        
        # Check last rank (rank 1, index 7)
        for col in range(8):
            piece = board_state[7, col]
            if piece in ['white_pawn', 'black_pawn']:
                errors.append(f"Pawn on rank 1 at file {chr(ord('a') + col)}")
        
        return errors
    
    def get_starting_position_fen(self) -> str:
        """Return FEN for standard chess starting position"""
        return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    def create_board_from_predictions(self, occupancy_predictions: np.ndarray,
                                    piece_predictions: List[str],
                                    piece_positions: List[Tuple[int, int]]) -> np.ndarray:
        """
        Create board state from model predictions
        
        Args:
            occupancy_predictions: 8x8 binary array (0=empty, 1=occupied)
            piece_predictions: List of piece class names for occupied squares
            piece_positions: List of (row, col) positions for piece predictions
            
        Returns:
            8x8 board state array
        """
        board_state = np.full((8, 8), 'empty', dtype=object)
        
        # Map piece predictions to board positions
        piece_idx = 0
        for row in range(8):
            for col in range(8):
                if occupancy_predictions[row, col] == 1:  # Occupied square
                    if piece_idx < len(piece_predictions):
                        board_state[row, col] = piece_predictions[piece_idx]
                        piece_idx += 1
        
        return board_state

def test_fen_generator():
    """Test the FEN generator with sample data"""
    generator = FENGenerator()
    
    print("Testing FEN Generator")
    print("=" * 50)
    
    # Test 1: Starting position
    starting_fen = generator.get_starting_position_fen()
    print(f"Starting position FEN: {starting_fen}")
    
    # Test 2: Convert FEN to board and back
    board_state, fen_info = generator.fen_to_board(starting_fen)
    print(f"Board shape: {board_state.shape}")
    print(f"FEN info: {fen_info}")
    
    # Test 3: Convert board back to FEN
    reconstructed_fen = generator.board_to_fen(board_state)
    print(f"Reconstructed FEN: {reconstructed_fen}")
    
    # Test 4: Validate FEN
    is_valid, errors = generator.validate_fen(starting_fen)
    print(f"Starting position valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    
    # Test 5: Create sample board from predictions
    print("\nTesting prediction integration:")
    
    # Sample occupancy predictions (random for testing)
    occupancy = np.random.choice([0, 1], size=(8, 8), p=[0.7, 0.3])
    
    # Sample piece predictions
    piece_classes = list(PIECE_TO_FEN.keys())
    num_occupied = np.sum(occupancy)
    piece_predictions = np.random.choice(piece_classes, size=num_occupied).tolist()
    
    # Get positions of occupied squares
    occupied_positions = [(r, c) for r in range(8) for c in range(8) if occupancy[r, c] == 1]
    
    # Create board state
    board_state = generator.create_board_from_predictions(occupancy, piece_predictions, occupied_positions)
    
    # Generate FEN
    fen = generator.board_to_fen(board_state)
    print(f"Generated FEN from predictions: {fen}")
    
    # Validate
    is_valid, errors = generator.validate_fen(fen)
    print(f"Generated FEN valid: {is_valid}")
    if errors:
        print(f"Validation errors: {errors}")

if __name__ == "__main__":
    test_fen_generator()
