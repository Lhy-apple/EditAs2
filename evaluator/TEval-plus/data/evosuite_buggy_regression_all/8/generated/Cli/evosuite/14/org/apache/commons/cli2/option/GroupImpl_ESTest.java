/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 01:57:56 GMT 2023
 */

package org.apache.commons.cli2.option;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Locale;
import org.apache.commons.cli2.DisplaySetting;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.WriteableCommandLine;
import org.apache.commons.cli2.commandline.WriteableCommandLineImpl;
import org.apache.commons.cli2.option.ArgumentImpl;
import org.apache.commons.cli2.option.DefaultOption;
import org.apache.commons.cli2.option.GroupImpl;
import org.apache.commons.cli2.option.PropertyOption;
import org.apache.commons.cli2.option.SourceDestArgument;
import org.apache.commons.cli2.validation.ClassValidator;
import org.apache.commons.cli2.validation.FileValidator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.System;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class GroupImpl_ESTest extends GroupImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "h>A~zl/SR#,={1w", "h>A~zl/SR#,={1w", 95, 95);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      writeableCommandLineImpl0.getSwitch((Option) groupImpl0);
      assertEquals(95, groupImpl0.getMinimum());
      assertEquals(95, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      DisplaySetting displaySetting0 = mock(DisplaySetting.class, new ViolatedAssumptionAnswer());
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "DISPLAY_GROUP_ARGUMENT", "DISPLAY_GROUP_ARGUMENT", (-321), (-321));
      StringBuffer stringBuffer0 = new StringBuffer();
      Comparator<DefaultOption> comparator0 = (Comparator<DefaultOption>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      groupImpl0.appendUsage(stringBuffer0, displaySetting0.ALL, comparator0);
      assertEquals("DISPLAY_GROUP_ARGUMENT ()", stringBuffer0.toString());
      assertEquals(25, stringBuffer0.length());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "H8>/FvzJdMx+F%6g}\"[", "H8>/FvzJdMx+F%6g}\"[", 0, 3309);
      groupImpl0.getAnonymous();
      assertEquals(0, groupImpl0.getMinimum());
      assertEquals(3309, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "9f[q}\"nFHVj>,4", "mv%eL_~kYoT>+", 0, 137);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      linkedList0.add(groupImpl0);
      // Undeclared exception!
      try { 
        groupImpl0.validate(writeableCommandLineImpl0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "H8>/FvzJdMx+F%6g}\"[", "H8>/FvzJdMx+F%6g}\"[", 0, 3309);
      int int0 = groupImpl0.getMaximum();
      assertEquals(0, groupImpl0.getMinimum());
      assertEquals(3309, int0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "Passes properties and values to the application", 44, 44);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "-D");
      assertEquals(1, linkedList0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "Passes properties and values to the application", 44, 44);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "Passes properties and values to the application");
      assertEquals(1, linkedList0.size());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption("", "", 4349);
      linkedList0.push(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Unexpected.token", "Unexpected.token", 4349, 4349);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "Unexpected.token");
      assertEquals(1, linkedList0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "Passes properties and values to the application", (-555), 0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "-D");
      assertEquals(0, groupImpl0.getMaximum());
      assertEquals((-555), groupImpl0.getMinimum());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      FileValidator fileValidator0 = FileValidator.getExistingFileInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("^@ )X)g,aO", "^@ )X)g,aO", 0, 0, 'p', '{', fileValidator0, "eAl_Bu8}j", linkedList0, 0);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0);
      linkedList0.add((ArgumentImpl) sourceDestArgument0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, "", (-1632), (-1632));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "org.apache.commons.cli2.option.OptionImpl");
      assertEquals(0, linkedList0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      LinkedList<String> linkedList1 = new LinkedList<String>();
      linkedList1.offer("Passes properties and values to the application");
      ListIterator<String> listIterator0 = linkedList1.listIterator();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", 4, 4);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertEquals(4, groupImpl0.getMinimum());
      assertEquals(4, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      System.setCurrentTimeMillis(1919L);
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      FileValidator fileValidator0 = FileValidator.getExistingFileInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("^@ )X)g,aO", "^@ )X)g,aO", 0, 0, 'p', '{', fileValidator0, "eAl_Bu8}j", linkedList0, 0);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0);
      linkedList0.add((ArgumentImpl) sourceDestArgument0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, "", (-1613), (-1613));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      assertEquals(0, linkedList0.size());
      
      SourceDestArgument sourceDestArgument1 = new SourceDestArgument(sourceDestArgument0, sourceDestArgument0, '\u0000', '\u0000', "1p\"3NmB?807%q4w", linkedList0);
      ListIterator<ArgumentImpl> listIterator0 = (ListIterator<ArgumentImpl>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(listIterator0).hasNext();
      groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, (ListIterator) listIterator0);
      groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, (String) null);
      groupImpl0.validate(writeableCommandLineImpl0);
      ListIterator<String> listIterator1 = (ListIterator<String>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true, false, false).when(listIterator1).hasNext();
      doReturn("").when(listIterator1).next();
      doReturn("--").when(listIterator1).previous();
      groupImpl0.process(writeableCommandLineImpl0, listIterator1);
      assertEquals((-1613), groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption("", "", 0);
      linkedList0.add(propertyOption0);
      LinkedList<String> linkedList1 = new LinkedList<String>();
      linkedList1.offer("");
      ListIterator<String> listIterator0 = linkedList1.listIterator();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", 0, 0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      try { 
        groupImpl0.process(writeableCommandLineImpl0, listIterator0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unexpected  while processing 
         //
         verifyException("org.apache.commons.cli2.option.PropertyOption", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption("", "", 4);
      PropertyOption propertyOption1 = new PropertyOption();
      linkedList0.addFirst(propertyOption1);
      linkedList0.add(propertyOption0);
      LinkedList<String> linkedList1 = new LinkedList<String>();
      linkedList1.offer("Passes properties and values to the application");
      ListIterator<String> listIterator0 = linkedList1.listIterator();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", 4, 4);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertFalse(listIterator0.hasNext());
      assertTrue(listIterator0.hasPrevious());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      LinkedList<String> linkedList1 = new LinkedList<String>();
      linkedList1.offer("|C\"!+0]fE3dVgYz/");
      ListIterator<String> listIterator0 = linkedList1.listIterator();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "|C\"!+0]fE3dVgYz/", "|C\"!+0]fE3dVgYz/", (-3324), (-3324));
      PropertyOption propertyOption0 = new PropertyOption("|C\"!+0]fE3dVgYz/", "|C\"!+0]fE3dVgYz/", (-3324));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList1);
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertEquals((-3324), groupImpl0.getMinimum());
      assertEquals((-3324), groupImpl0.getMaximum());
      assertTrue(listIterator0.hasNext());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      FileValidator fileValidator0 = FileValidator.getExistingFileInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("^@ )X)g,aO", "^@ )X)g,aO", 0, 0, 'p', '{', fileValidator0, "eAl_Bu8}j", linkedList0, 0);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0);
      linkedList0.add((ArgumentImpl) sourceDestArgument0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, "", (-1632), (-1632));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      ListIterator<String> listIterator0 = (ListIterator<String>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true, false, true, false, false).when(listIterator0).hasNext();
      doReturn("", "--").when(listIterator0).next();
      doReturn("--", "--").when(listIterator0).previous();
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertEquals((-1632), groupImpl0.getMaximum());
      assertEquals((-1632), groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", 93, 93);
      linkedList0.add(groupImpl0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      // Undeclared exception!
      try { 
        groupImpl0.validate(writeableCommandLineImpl0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "Passes properties and values to the application", 44, 44);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      try { 
        groupImpl0.validate(writeableCommandLineImpl0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing option Passes properties and values to the application
         //
         verifyException("org.apache.commons.cli2.option.GroupImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      ClassValidator classValidator0 = new ClassValidator();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("@1fk0", "R", 2592, 2592, '.', 'F', classValidator0, "R", linkedList0, 2592);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(argumentImpl0, linkedList0);
      PropertyOption propertyOption0 = new PropertyOption("R", "x6i-m.=f19d#~`s", 115);
      LinkedList<PropertyOption> linkedList1 = new LinkedList<PropertyOption>();
      linkedList1.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList1, "R", "R", 2592, 1);
      writeableCommandLineImpl0.addOption(propertyOption0);
      try { 
        groupImpl0.validate(writeableCommandLineImpl0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing option R
         //
         verifyException("org.apache.commons.cli2.option.GroupImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "Passes properties and values to the application", (-512), (-512));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addOption(propertyOption0);
      try { 
        groupImpl0.validate(writeableCommandLineImpl0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unexpected -D while processing Passes properties and values to the application
         //
         verifyException("org.apache.commons.cli2.option.GroupImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Unexpected.tGken", "Unexpected.tGken", 0, 0);
      String string0 = groupImpl0.toString();
      assertEquals(0, groupImpl0.getMaximum());
      assertEquals("[Unexpected.tGken ()]", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, 0, 0);
      linkedList0.add(groupImpl0);
      // Undeclared exception!
      groupImpl0.toString();
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, 60, 60);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      try { 
        groupImpl0.validate(writeableCommandLineImpl0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing option 
         //
         verifyException("org.apache.commons.cli2.option.GroupImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", 935, 935);
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      linkedList0.addLast(propertyOption0);
      linkedList0.add(propertyOption0);
      String string0 = groupImpl0.toString();
      assertEquals(" (-D<property>=<value>|-D<property>=<value>)", string0);
      assertEquals(935, groupImpl0.getMaximum());
      assertEquals(935, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      ClassValidator classValidator0 = new ClassValidator();
      LinkedList<List<String>> linkedList1 = new LinkedList<List<String>>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("OP|VF0(OV:~On\" )", "qyyWn]!fs", 2833, 2833, '0', '\'', classValidator0, "", linkedList1, 2833);
      linkedList0.add(argumentImpl0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "org.apache.commons.cli2.validation.EnumValidator", "org.apache.commons.cli2.validation.EnumValidator", (-1), (-7));
      groupImpl0.toString();
      assertEquals((-7), groupImpl0.getMaximum());
      assertEquals((-1), groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Un#jexpcted.oe", "Un#jexpcted.oe", 0, 0);
      LinkedHashSet<GroupImpl> linkedHashSet0 = new LinkedHashSet<GroupImpl>();
      List list0 = groupImpl0.helpLines(0, linkedHashSet0, (Comparator) null);
      assertEquals(0, groupImpl0.getMinimum());
      assertEquals(0, groupImpl0.getMaximum());
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      DisplaySetting displaySetting0 = mock(DisplaySetting.class, new ViolatedAssumptionAnswer());
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "PBCtEHlcvv[hT", "PBCtEHlcvv[hT", (-3074), 1094);
      List list0 = groupImpl0.helpLines((-1025), displaySetting0.ALL, (Comparator) null);
      assertEquals(1094, groupImpl0.getMaximum());
      assertEquals((-3074), groupImpl0.getMinimum());
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      DisplaySetting displaySetting0 = mock(DisplaySetting.class, new ViolatedAssumptionAnswer());
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Unexpected.token", "Unexpected.token", (-340), 1084);
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      linkedList0.add(propertyOption0);
      Comparator<Locale.LanguageRange> comparator0 = (Comparator<Locale.LanguageRange>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      List list0 = groupImpl0.helpLines((-340), displaySetting0.ALL, comparator0);
      assertEquals(1084, groupImpl0.getMaximum());
      assertEquals((-340), groupImpl0.getMinimum());
      assertEquals(2, list0.size());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption(" (", " (", 927);
      linkedList0.addFirst(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "-D", (-698), (-164));
      groupImpl0.findOption("Passes properties and values to the application");
      assertEquals((-164), groupImpl0.getMaximum());
      assertEquals("Passes properties and values to the application", groupImpl0.getPreferredName());
      assertEquals("-D", groupImpl0.getDescription());
      assertEquals((-698), groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, 0, 0);
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      linkedList0.addFirst(propertyOption0);
      Option option0 = groupImpl0.findOption("-D");
      assertNotNull(option0);
      assertEquals(0, groupImpl0.getMaximum());
      assertEquals(0, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "h>A~zP/SSR#,={1w", "h>A~zP/SSR#,={1w", 86, 86);
      linkedList0.add(groupImpl0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      // Undeclared exception!
      try { 
        groupImpl0.defaults(writeableCommandLineImpl0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      ClassValidator classValidator0 = new ClassValidator();
      LinkedList<List<String>> linkedList1 = new LinkedList<List<String>>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("OP|VF0(OV:~On\" )", "qyyWn]!fs", 2833, 2833, '0', '\'', classValidator0, "", linkedList1, 2833);
      linkedList0.add(argumentImpl0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "org.apache.commons.cli2.validation.EnumValidator", "org.apache.commons.cli2.validation.EnumValidator", (-1), (-7));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(argumentImpl0, (List) null);
      groupImpl0.defaults(writeableCommandLineImpl0);
      assertEquals((-1), groupImpl0.getMinimum());
      assertEquals((-7), groupImpl0.getMaximum());
  }
}
