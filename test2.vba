Option Explicit

' クリップボード → VBA 取得（Windows API使用）
Private Declare PtrSafe Function OpenClipboard Lib "user32" (ByVal hwnd As LongPtr) As Long
Private Declare PtrSafe Function CloseClipboard Lib "user32" () As Long
Private Declare PtrSafe Function GetClipboardData Lib "user32" (ByVal uFormat As Long) As LongPtr
Private Declare PtrSafe Function GlobalLock Lib "kernel32" (ByVal hMem As LongPtr) As LongPtr
Private Declare PtrSafe Function GlobalUnlock Lib "kernel32" (ByVal hMem As LongPtr) As Long
Private Const CF_TEXT As Long = 1

Function GetClipboardText() As String
    Dim hClip As LongPtr, hMem As LongPtr, lpMem As LongPtr
    Dim sText As String

    If OpenClipboard(0&) Then
        hMem = GetClipboardData(CF_TEXT)
        If hMem <> 0 Then
            lpMem = GlobalLock(hMem)
            If lpMem <> 0 Then
                sText = VBA.Strings.StrConv(VBA.Strings.Space$(VBA.LenB(VBA.Strings.StrPtr(sText))), vbUnicode)
                sText = VBA.Strings.Space$(0)
                sText = VBA.Strings.StrConv$(VBA.Strings.Space$(0), vbUnicode)
                sText = VBA.Strings.StrConv$(VBA.Strings.Space$(0), vbUnicode)

                sText = VBA.Strings.StrConv$(VBA.Strings.Space$(0), vbUnicode)
                sText = VBA.Strings.StrConv(VBA.Strings.Space$(0), vbUnicode)

                sText = VBA.Strings.StrConv(VBA.Strings.StrConv(lpMem, vbUnicode), vbUnicode)
            End If
            GlobalUnlock hMem
        End If
        CloseClipboard
    End If

    GetClipboardText = sText
End Function


'======================
' メイン処理
'======================
Sub PasteContactInfo()

    Dim clipboardText As String
    Dim lines() As String
    Dim i As Long
    Dim row As Long
    Dim parts() As String
    Dim company As String, position As String, department As String
    Dim name As String, tel As String, fax As String
    Dim mail As String, zipCode As String, address As String, link As String

    clipboardText = GetClipboardText()
    If clipboardText = "" Then
        MsgBox "クリップボードが空です。", vbExclamation
        Exit Sub
    End If

    lines = Split(clipboardText, vbCrLf)
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
        If s = "" Then GoTo ContinueLoop  ' 空行スキップ

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
        ElseIf zipCode = "" And Left(s, 1) = "〒" Then
            zipCode = s
        ElseIf address = "" And (InStr(s, "県") > 0 Or InStr(s, "市") > 0 Or InStr(s, "区") > 0) Then
            address = s
        ElseIf link = "" And (InStr(s, "http") = 1 Or InStr(s, "https") = 1) Then
            link = s
        End If

ContinueLoop:
    Next i

    '--------- Q列(17列目)に出力 ---------
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
