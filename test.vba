' クリップボード取得用のAPI宣言
#If VBA7 Then
    Private Declare PtrSafe Function OpenClipboard Lib "user32" (ByVal hwnd As LongPtr) As Long
    Private Declare PtrSafe Function GetClipboardData Lib "user32" (ByVal uFormat As Long) As LongPtr
    Private Declare PtrSafe Function CloseClipboard Lib "user32" () As Long
    Private Declare PtrSafe Function GlobalLock Lib "kernel32" (ByVal hMem As LongPtr) As LongPtr
    Private Declare PtrSafe Function GlobalUnlock Lib "kernel32" (ByVal hMem As LongPtr) As Long
#Else
    Private Declare Function OpenClipboard Lib "user32" (ByVal hwnd As Long) As Long
    Private Declare Function GetClipboardData Lib "user32" (ByVal uFormat As Long) As Long
    Private Declare Function CloseClipboard Lib "user32" () As Long
    Private Declare Function GlobalLock Lib "kernel32" (ByVal hMem As Long) As Long
    Private Declare Function GlobalUnlock Lib "kernel32" (ByVal hMem As Long) As Long
#End If

Const CF_TEXT = 1

Function GetClipboardText() As String
    Dim hClipMemory As LongPtr
    Dim lpClipMemory As LongPtr
    Dim ClipText As String

    If OpenClipboard(0&) Then
        hClipMemory = GetClipboardData(CF_TEXT)
        If hClipMemory <> 0 Then
            lpClipMemory = GlobalLock(hClipMemory)
            If lpClipMemory <> 0 Then
                ClipText = VBA.Strings.StrConv(VBA.Strings.StrConv( _
                    VBA.Strings.Space$(65535), vbFromUnicode _
                ), vbUnicode)
                ClipText = VBA.Strings.StrConv( _
                    VBA.Strings.StrConv(StrPtr(lpClipMemory), vbUnicode), vbFromUnicode)
                GlobalUnlock hClipMemory
            End If
        End If
        CloseClipboard
    End If

    GetClipboardText = ClipText
End Function


' ======= メイン処理 =======
Sub PasteBusinessCardData()

    Dim text As String
    Dim lines As Variant
    Dim i As Long

    ' クリップボードの内容を取得
    text = GetClipboardText()

    If text = "" Then
        MsgBox "クリップボードが空です"
        Exit Sub
    End If

    ' 行ごとに分割
    lines = Split(text, vbCrLf)

    ' ===== データの割り当て =====
    ' 想定フォーマット：
    ' 1:会社名
    ' 2:役職 所属
    ' 3:氏名
    '
    ' 空行
    ' 電話番号
    ' FAX（無い場合はスキップ）
    ' メールアドレス
    '
    ' [住所]
    ' 郵便番号
    ' 住所
    ' 
    ' リンク
    ' ＝＝＝＝＝＝＝＝＝＝

    Dim row As Long: row = ActiveCell.Row

    '---- 上部情報 ----
    Cells(row, 1).Value = lines(0) '会社名

    ' 役職と所属を分割（スペース区切り）
    Dim pos_dept() As String
    pos_dept = Split(lines(1), " ")

    '役職
    Cells(row, 3).Value = pos_dept(0)

    '所属（役職が1語の場合）
    If UBound(pos_dept) >= 1 Then
        Cells(row, 2).Value = pos_dept(1)
    End If

    '氏名
    Cells(row, 7).Value = lines(2)

    '---- 下部情報 ----
    Dim idx As Long: idx = 4   '4行目から電話番号が始まる

    '空行をスキップ
    Do While Trim(lines(idx)) = ""
        idx = idx + 1
    Loop

    Cells(row, 6).Value = lines(idx) '電話番号
    idx = idx + 1

    'FAXがある場合
    If InStr(lines(idx), "@") = 0 Then 'メールでなければFAX
        idx = idx + 1
    End If

    'メールアドレス
    Cells(row, 8).Value = lines(idx)
    idx = idx + 1

    ' "[住所]" 行をスキップ
    If InStr(lines(idx), "住所") > 0 Then
        idx = idx + 1
    End If

    ' 郵便番号
    Cells(row, 4).Value = lines(idx)
    idx = idx + 1

    ' 住所
    Cells(row, 5).Value = lines(idx)

    MsgBox "貼り付け完了！"

End Sub
