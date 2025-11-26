Sub PasteContactInfo()

    Dim clipboardText As String
    Dim lines() As String
    Dim i As Long
    Dim row As Long
    Dim parts() As String
    Dim company As String, position As String, department As String
    Dim name As String, tel As String, fax As String
    Dim mail As String, zipCode As String, address As String, link As String
    
    ' クリップボードから取得
    clipboardText = GetClipboardText()
    lines = Split(clipboardText, vbCrLf)
    
    ' 貼り付け開始行（アクティブセルの行）
    row = ActiveCell.Row
    
    ' 初期化
    company = ""
    position = ""
    department = ""
    name = ""
    tel = ""
    fax = ""
    mail = ""
    zipCode = ""
    address = ""
    link = ""
    
    '--------- データ解析 ---------
    For i = 0 To UBound(lines)
        
        Dim s As String
        s = Trim(lines(i))
        
        If s = "" Then GoTo ContinueLoop   ' 空行スキップ（構文エラー修正）

        If company = "" Then
            company = s
        ElseIf InStr(s, " ") > 0 And position = "" Then
            parts = Split(s, " ")
            position = parts(0)
            If UBound(parts) >= 1 Then department = parts(1)
        ElseIf name = "" Then
            name = s
        ElseIf tel = "" And (InStr(s, "0") = 1 Or InStr(s, "-") > 0) Then
            tel = s
        ElseIf fax = "" And InStr(UCase(s), "FAX") > 0 Then
            fax = Replace(UCase(s), "FAX", "")
            fax = Trim(Replace(fax, "：", ""))
        ElseIf mail = "" And InStr(s, "@") > 0 Then
            mail = s
        ElseIf InStr(s, "[住所]") > 0 Then
            ' 住所ラベルはスキップ
        ElseIf zipCode = "" And (Left(s, 1) = "〒" Or IsNumeric(Replace(s, "-", ""))) Then
            zipCode = s
        ElseIf address = "" And (InStr(s, "県") > 0 Or InStr(s, "市") > 0 Or InStr(s, "区") > 0) Then
            address = s
        ElseIf link = "" And (InStr(s, "http") = 1 Or InStr(s, "https") = 1) Then
            link = s
        End If

ContinueLoop:
    Next i
    
    '--------- Excel(Q列〜)に出力 ---------
    With Cells(row, "Q")
        .Value = company
        .Offset(0, 1).Value = department
        .Offset(0, 2).Value = position
        .Offset(0, 3).Value = zipCode
        .Offset(0, 4).Value = address
        .Offset(0, 5).Value = tel
        .Offset(0, 6).Value = name
        .Offset(0, 7).Value = mail
        .Offset(0, 8).Value = link
        .Offset(0, 9).Value = fax
    End With

End Sub


'------------- クリップボード取得関数（Windows専用） -------------
Public Function GetClipboardText() As String
    Dim DataObj As Object
    Set DataObj = CreateObject("MSForms.DataObject")
    On Error GoTo ErrHandler
    DataObj.GetFromClipboard
    GetClipboardText = DataObj.GetText(1)
    Exit Function
ErrHandler:
    GetClipboardText = ""
End Function
