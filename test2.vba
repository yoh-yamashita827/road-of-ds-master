Option Explicit

Sub PasteBusinessCard_Full()

    Dim DataObj As New MSForms.DataObject
    Dim clipboardText As String
    Dim lines() As String
    Dim row As Long, idx As Long
    Dim parts() As String
    
    Dim company As String, position As String, department As String
    Dim name As String, tel As String, fax As String
    Dim mail As String, zipCode As String, address As String, link As String
    
    ' 初期化
    company = "": position = "": department = ""
    name = "": tel = "": fax = ""
    mail = "": zipCode = "": address = "": link = ""
    
    ' クリップボード取得
    On Error GoTo ErrClip
    DataObj.GetFromClipboard
    clipboardText = DataObj.GetText(1)
    On Error GoTo 0
    
    If clipboardText = "" Then
        MsgBox "クリップボードが空です。", vbExclamation
        Exit Sub
    End If
    
    ' 改行で分割
    lines = Split(clipboardText, vbCrLf)
    row = ActiveCell.Row
    idx = 0
    
    '-----------------------
    ' 会社名
    '-----------------------
    Do While idx <= UBound(lines) And Trim(lines(idx)) = ""
        idx = idx + 1
    Loop
    If idx <= UBound(lines) Then
        company = Trim(lines(idx))
        idx = idx + 1
    End If
    
    '-----------------------
    ' 役職・所属
    '-----------------------
    Do While idx <= UBound(lines) And Trim(lines(idx)) = ""
        idx = idx + 1
    Loop
    If idx <= UBound(lines) Then
        parts = Split(Replace(lines(idx), "　", " "))
        position = parts(0)
        If UBound(parts) >= 1 Then department = parts(1)
        idx = idx + 1
    End If
    
    '-----------------------
    ' 氏名
    '-----------------------
    Do While idx <= UBound(lines) And Trim(lines(idx)) = ""
        idx = idx + 1
    Loop
    If idx <= UBound(lines) Then
        name = Trim(lines(idx))
        idx = idx + 1
    End If
    
    '-----------------------
    ' 電話番号
    '-----------------------
    Do While idx <= UBound(lines) And Trim(lines(idx)) = ""
        idx = idx + 1
    Loop
    If idx <= UBound(lines) Then
        tel = Trim(lines(idx))
        idx = idx + 1
    End If
    
    '-----------------------
    ' FAX またはメール
    '-----------------------
    Do While idx <= UBound(lines) And Trim(lines(idx)) = ""
        idx = idx + 1
    Loop
    If idx <= UBound(lines) Then
        If InStr(lines(idx), "@") = 0 Then
            fax = Trim(lines(idx))
            idx = idx + 1
        End If
    End If
    
    '-----------------------
    ' メール
    '-----------------------
    Do While idx <= UBound(lines) And Trim(lines(idx)) = ""
        idx = idx + 1
    Loop
    If idx <= UBound(lines) Then
        mail = Trim(lines(idx))
        idx = idx + 1
    End If
    
    '-----------------------
    ' [住所] 行スキップ
    '-----------------------
    Do While idx <= UBound(lines) And (Trim(lines(idx)) = "" Or InStr(lines(idx), "住所") > 0)
        idx = idx + 1
    Loop
    
    '-----------------------
    ' 郵便番号
    '-----------------------
    If idx <= UBound(lines) Then
        zipCode = Trim(lines(idx))
        idx = idx + 1
    End If
    
    '-----------------------
    ' 住所
    '-----------------------
    Do While idx <= UBound(lines) And Trim(lines(idx)) = ""
        idx = idx + 1
    Loop
    If idx <= UBound(lines) Then
        address = Trim(lines(idx))
        idx = idx + 1
    End If
    
    '-----------------------
    ' リンク（あれば）
    '-----------------------
    Do While idx <= UBound(lines) And Trim(lines(idx)) = ""
        idx = idx + 1
    Loop
    If idx <= UBound(lines) Then
        link = Trim(lines(idx))
        idx = idx + 1
    End If
    
    '-----------------------
    ' Excel Q列(17列目)に出力
    '-----------------------
    With Cells(row, 17)
        .Value = company           ' Q列
        .Offset(0, 1).Value = department  ' R列
        .Offset(0, 2).Value = position    ' S列
        .Offset(0, 3).Value = zipCode     ' T列
        .Offset(0, 4).Value = address     ' U列
        .Offset(0, 5).Value = tel         ' V列
        .Offset(0, 6).Value = name        ' W列
        .Offset(0, 7).Value = mail        ' X列
        .Offset(0, 8).Value = fax         ' Y列
        .Offset(0, 9).Value = link        ' Z列
    End With
    
    MsgBox "名刺情報をQ列から展開しました！", vbInformation
    Exit Sub

ErrClip:
    MsgBox "クリップボードから取得できませんでした。" & vbCrLf & _
           "Microsoft Forms 2.0 Object Library が参照に追加されているか確認してください。", vbExclamation

End Sub
