/*
 * Copyright (c) 2012-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SETTINGS_DELEGATE_H_
#define OSKAR_SETTINGS_DELEGATE_H_

#include <QStyledItemDelegate>

class QModelIndex;
class QWidget;

namespace oskar {

class SettingsDelegate : public QStyledItemDelegate
{
    Q_OBJECT

 public:
    SettingsDelegate(QWidget* view, QObject* parent = 0);

    QWidget* createEditor(QWidget* parent, const QStyleOptionViewItem& option,
            const QModelIndex& index) const;

    bool editorEvent(QEvent* event, QAbstractItemModel* model,
            const QStyleOptionViewItem& option, const QModelIndex& index);

    void paint(QPainter* painter, const QStyleOptionViewItem& option,
            const QModelIndex &index) const;

    void setEditorData(QWidget* editor, const QModelIndex& index) const;

    void setModelData(QWidget* editor, QAbstractItemModel* model,
            const QModelIndex& index) const;

    void updateEditorGeometry(QWidget* editor,
            const QStyleOptionViewItem& option,
            const QModelIndex& index) const;

private slots:
    void commitAndCloseEditor(int index);

private:
    QWidget* view_;
};

} /* namespace oskar */

#endif /* OSKAR_SETTINGS_DELEGATE_H_ */
