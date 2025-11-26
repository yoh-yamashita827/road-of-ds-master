'--------------------------------------------------------
' Windows Excel 用：名刺情報 → Q列から各列に展開
'--------------------------------------------------------

' クリップボード取得API（Windows）
#If VBA7 Then
    Private Declare PtrSafe Function OpenClipboard Lib "user32" (ByVal hwnd As LongPtr) As Long
    Private Declare PtrSafe Function CloseClipboard Lib "user32" () As Long
    Private Declare PtrSafe Function GetClipboardData Lib "user32" (ByVal uFormat As Long) As LongPtr
    Private Declare PtrSafe Function GlobalLock Lib "kernel32" (ByVal hMem As LongPtr) As LongPtr
    Private Declare PtrSafe Function GlobalUnlock Lib "kernel32" (ByVal hMem As LongPtr) As Long
#Else
    Private Declare Function OpenClipboard Lib "user32" (ByVal hwnd As Long) As Long
    Private Declare Function CloseClipboard Lib "user32" () As Long
    Private Declare Function GetClipboardData Lib "user32" (ByVal uFormat As Long) As Long
    Private Declare Function GlobalLock Lib "kernel32" (ByVal hMem As Long) As Long
    Private Declare Function GlobalUnlock Lib "kernel32" (ByVal hMem As Long) As Long
#End If

Const CF_TEXT = 1


'--- クリップボードの文字列を取得 ---
Function GetClipboardTextWindows() As String
    Dim hClip As LongPtr
    Dim pText As LongPtr
    Dim s As String

    If OpenClipboard(0) Then
        hClip = GetClipboardData(CF_TEXT)
        If hClip <> 0 Then
            pText = GlobalLock(hClip)
            If pText <> 0 Then
                s = StrConv(StrConv(pText, vbUnicode), vbFromUnicode)
                GlobalUnlock hClip
            End If
        End If
        CloseClipboard
    End If

    GetClipboardTextWindows = s
End Function


'--------------------------------------------------------
' メイン処理：Q列から自動展開
'--------------------------------------------------------
Sub PasteBusinessCard_Q()

    Dim text As String
    Dim lines As Variant
    Dim idx As Long
    Dim row As Long
    
    ' Q列 = 17列目
    Const START_COL As Long = 17

    text = GetClipboardTextWindows()

    If Trim(text) = "" Then
        MsgBox "クリップボードが空です。"
        Exit Sub
    End If

    lines = Split(text, vbCrLf)
    row = ActiveCell.Row
    idx = 0

    '--------------------------------------------------------
    ' 基本3行：会社名 / 役職 所属 / 氏名
    '--------------------------------------------------------
    Cells(row, START_COL).Value = Trim(lines(idx)) '会社名
    idx = idx + 1

    '役職 所属
    Dim posDept As Variant
    Dim line2 As String
    line2 = Replace(lines(idx), "　", " ") '全角スペース→半角
    posDept = Split(Trim(line2), " ")
    
    Cells(row, START_COL + 2).Value = posDept(0) '役職（Q+2 = S列）
    If UBound(posDept) >= 1 Then
        Cells(row, START_COL + 1).Value = posDept(1) '所属（Q+1 = R列）
    End If
    idx = idx + 1

    '氏名
    Cells(row, START_COL + 6).Value = Trim(lines(idx)) 'W列
    idx = idx + 1

    '--------------------------------------------------------
    ' 空行をスキップ
    '--------------------------------------------------------
    Do While idx <= UBound(lines) And Trim(lines[idx]) = ""
        idx = idx + 1
    Loop

    '--------------------------------------------------------
    ' 電話番号
    '--------------------------------------------------------
    Cells(row, START_COL + 5).Value = Trim(lines(idx)) 'V列
    idx = idx + 1

    '--------------------------------------------------------
    ' FAX があれば入れる（なければスキップ）
    '--------------------------------------------------------
    If idx <= UBound(lines) And InStr(lines(idx), "@") = 0 Then
        Cells(row, START_COL + 8).Value = Trim(lines(idx)) 'Y列
        idx = idx + 1
    End If

    '--------------------------------------------------------
    ' メールアドレス
    '--------------------------------------------------------
    Cells(row, START_COL + 7).Value = Trim(lines(idx)) 'X列
    idx = idx + 1

    '--------------------------------------------------------
    ' "[住所]" をスキップ
    '--------------------------------------------------------
    Do While idx <= UBound(lines) And _
        (Trim(lines(idx)) = "" Or InStr(lines(idx), "住所") > 0)
        idx = idx + 1
    Loop

    '郵便番号
    Cells(row, START_COL + 3).Value = Trim(lines(idx)) 'T列
    idx = idx + 1

    '住所
    Cells(row, START_COL + 4).Value = Trim(lines(idx)) 'U列
    idx = idx + 1

    '--------------------------------------------------------
    ' リンク（Z列）
    '--------------------------------------------------------
    If idx <= UBound(lines) Then
        Cells(row, START_COL + 9).Value = Trim(lines[idx])
    End If

    MsgBox "取り込み完了！（Q列から展開しました）"

End Sub
