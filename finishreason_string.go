// Code generated by "stringer -type=FinishReason"; DO NOT EDIT.

package chat

import "strconv"

func _() {
	// An "invalid array index" compiler error signifies that the constant values have changed.
	// Re-run the stringer command to generate them again.
	var x [1]struct{}
	_ = x[FinishReasonStop-1]
	_ = x[FinishReasonLength-2]
	_ = x[FinishReasonUnknown-3]
}

const _FinishReason_name = "FinishReasonStopFinishReasonLengthFinishReasonUnknown"

var _FinishReason_index = [...]uint8{0, 16, 34, 53}

func (i FinishReason) String() string {
	i -= 1
	if i < 0 || i >= FinishReason(len(_FinishReason_index)-1) {
		return "FinishReason(" + strconv.FormatInt(int64(i+1), 10) + ")"
	}
	return _FinishReason_name[_FinishReason_index[i]:_FinishReason_index[i+1]]
}
