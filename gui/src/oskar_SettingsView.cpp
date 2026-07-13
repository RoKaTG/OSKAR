/*
 * Copyright (c) 2012-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "gui/oskar_SettingsView.h"
#include "gui/oskar_SettingsModel.h"
#include <QApplication>
#include <QMouseEvent>
#include <QMessageBox>
#include <QSettings>
#include <QScrollBar>

namespace oskar {

SettingsView::SettingsView(QWidget* parent) : QTreeView(parent)
{
    connect(this, SIGNAL(expanded(const QModelIndex&)),
            this, SLOT(resizeAfterExpand(const QModelIndex&)));
    connect(this, SIGNAL(collapsed(const QModelIndex&)),
            this, SLOT(updateAfterCollapsed(const QModelIndex&)));
    connect(qApp, SIGNAL(focusChanged(QWidget*, QWidget*)),
            this, SLOT(focusChanged(QWidget*, QWidget*)));
    setAlternatingRowColors(true);
    setUniformRowHeights(true);
#ifdef Q_OS_WIN
    setStyleSheet("\
        QTreeView {\
            show-decoration-selected: 0;\
            selection-background-color: transparent;\
        }\
    ");
#endif
}

void SettingsView::displayLabels()
{
    if (!model()) return;
    model()->setData(QModelIndex(), false, SettingsModel::DisplayKeysRole);
}

void SettingsView::displayKeys()
{
    if (!model()) return;
    model()->setData(QModelIndex(), true, SettingsModel::DisplayKeysRole);
}

void SettingsView::restoreExpanded(const QString& app)
{
    if (!model()) return;
    QSettings s;
    QStringList expanded =
            s.value("settings_view/expanded_items/" + app).toStringList();
    saveRestoreExpanded(QModelIndex(), expanded, 1);
}

void SettingsView::restorePosition()
{
    QSettings settings;
    QScrollBar* verticalScroll = verticalScrollBar();
    verticalScroll->setValue(settings.value("settings_view/position").toInt());
}

void SettingsView::saveExpanded(const QString& app)
{
    if (!model()) return;
    QSettings s;
    QStringList expanded;
    saveRestoreExpanded(QModelIndex(), expanded, 0);
    s.setValue("settings_view/expanded_items/" + app, expanded);
}

void SettingsView::savePosition()
{
    QSettings settings;
    settings.setValue("settings_view/position", verticalScrollBar()->value());
}

void SettingsView::showFirstLevel()
{
    expandToDepth(0);
    resizeColumnToContents(0);
    update();
}

void SettingsView::expandSettingsTree()
{
    expandAll();
    resizeColumnToContents(0);
    update();
}

void SettingsView::resizeAfterExpand(const QModelIndex& /*index*/)
{
    resizeColumnToContents(0);
    update();
}


void SettingsView::updateAfterCollapsed(const QModelIndex& /*index*/)
{
    update();
}

void SettingsView::focusChanged(QWidget* old, QWidget* now)
{
    if (!old && now && model())
    {
        // OSKAR has gained focus.
        // Check if the settings file has been modified more recently than the
        // last known modification date.
        model()->setData(QModelIndex(), QVariant(),
                SettingsModel::CheckExternalChangesRole);
    }
}

void SettingsView::fileReloaded()
{
    QMessageBox msgBox(this);
    msgBox.setWindowTitle(parentWidget()->windowTitle());
    msgBox.setIcon(QMessageBox::Information);
    msgBox.setText("The settings file was updated by another application.");
    msgBox.setInformativeText("It has now been re-loaded.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.exec();
}

void SettingsView::mouseDoubleClickEvent(QMouseEvent* event)
{
    QModelIndex index = indexAt(event->pos());
    QModelIndex sibling = index.sibling(index.row(), 1);
    if (model()->flags(sibling) & Qt::ItemIsEditable)
    {
        setCurrentIndex(sibling);
        edit(sibling, QAbstractItemView::DoubleClicked, event);
    }
    else
    {
        QTreeView::mouseDoubleClickEvent(event);
    }
}

void SettingsView::saveRestoreExpanded(const QModelIndex& parent,
        QStringList& list, int restore)
{
    if (!model()) return;
    for (int i = 0; i < model()->rowCount(parent); ++i)
    {
        QModelIndex idx = model()->index(i, 0, parent);
        QString key = idx.data(SettingsModel::KeyRole).toString();
        if (restore)
        {
            if (list.contains(key)) expand(idx);
        }
        else
        {
            if (isExpanded(idx)) list.append(key);
        }

        // Recursion.
        if (model()->rowCount(idx) > 0)
            saveRestoreExpanded(idx, list, restore);
    }
}

} // namespace oskar
