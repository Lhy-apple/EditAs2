/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:27:47 GMT 2023
 */

package org.apache.commons.cli2.option;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.text.NumberFormat;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Set;
import org.apache.commons.cli2.WriteableCommandLine;
import org.apache.commons.cli2.commandline.WriteableCommandLineImpl;
import org.apache.commons.cli2.option.ArgumentImpl;
import org.apache.commons.cli2.option.DefaultOption;
import org.apache.commons.cli2.option.GroupImpl;
import org.apache.commons.cli2.option.PropertyOption;
import org.apache.commons.cli2.option.SourceDestArgument;
import org.apache.commons.cli2.option.Switch;
import org.apache.commons.cli2.validation.DateValidator;
import org.apache.commons.cli2.validation.NumberValidator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class GroupImpl_ESTest extends GroupImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "i", "i", 114, 114, true);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      ListIterator<Object> listIterator0 = (ListIterator<Object>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true).when(listIterator0).hasNext();
      doReturn("i").when(listIterator0).next();
      doReturn(writeableCommandLineImpl0).when(listIterator0).previous();
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertEquals(114, groupImpl0.getMaximum());
      assertEquals(114, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      LinkedList<String> linkedList0 = new LinkedList<String>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, "+r`]JP#D2W~'),gf#", 0, 95, true);
      groupImpl0.getAnonymous();
      assertEquals(0, groupImpl0.getMinimum());
      assertEquals(95, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "}Ao/-^TSf|b~0]", "}Ao/-^TSf|b~0]", 0, (-945), false);
      int int0 = groupImpl0.getMaximum();
      assertEquals((-945), int0);
      assertEquals(0, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      DateValidator dateValidator0 = DateValidator.getDateInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("Passes properties and values to the application", "Passes properties and values to the application", 1441, 1441, ',', ',', dateValidator0, "DISPLAY_PARENT_ARGUMENT", linkedList0, 2479);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0);
      linkedList0.add((Object) sourceDestArgument0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "--", "Passes properties and values to the application", 1441, (-244), true);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(sourceDestArgument0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "org.apache.commons.cli2.commandline.WriteableCommandLineImpl");
      assertEquals(0, linkedList0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      PropertyOption propertyOption0 = new PropertyOption("org.apache.commons.cli2.option.GroupImpl", "DISPLAY_PARENT_ARGUMENT", 1441);
      linkedList0.add((Object) propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "DISPLAY_PARENT_ARGUMENT", "org.apache.commons.cli2.option.GroupImpl", 1441, 1441, false);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "org.apache.commons.cli2.option.GroupImpl");
      assertEquals(1, linkedList0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", (-1), (-1), true);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, (String) null);
      assertEquals((-1), groupImpl0.getMinimum());
      assertFalse(boolean0);
      assertEquals((-1), groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      PropertyOption propertyOption0 = new PropertyOption("org.apache.commons.cli2.option.GroupImpl", "DISPLAY_PARENT_ARGUMENT", 1441);
      linkedList0.add((Object) propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "DISPLAY_PARENT_ARGUMENT", "org.apache.commons.cli2.option.GroupImpl", 1441, 1441, false);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "|");
      assertEquals(1, linkedList0.size());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      PropertyOption propertyOption0 = new PropertyOption("", ":xg?5K*GLQ", (-14));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      linkedList0.add((Object) propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "-D", "2rEdrh", (-16), (-133), false);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, ":xg?5K*GLQ");
      assertEquals(1, linkedList0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      LinkedList<GroupImpl> linkedList1 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "<}#0}C;)6f(6g`z", "<}#0}C;)6f(6g`z", 22, (-1), false);
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList1);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "-D");
      assertEquals((-1), groupImpl0.getMaximum());
      assertFalse(boolean0);
      assertEquals(22, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      DateValidator dateValidator0 = DateValidator.getDateInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("Passes properties and values to the application", "Passes properties and values to the application", 1441, 1441, ',', ',', dateValidator0, "DISPLAY_PARENT_ARGUMENT", linkedList0, 2479);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "--", "Passes properties and values to the application", 1441, (-244), true);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(sourceDestArgument0, linkedList0);
      ListIterator<String> listIterator0 = (ListIterator<String>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(listIterator0).hasNext();
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertEquals(1441, groupImpl0.getMinimum());
      assertEquals("--", groupImpl0.getPreferredName());
      assertEquals((-244), groupImpl0.getMaximum());
      assertEquals("Passes properties and values to the application", groupImpl0.getDescription());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", (-1), (-1), true);
      ListIterator<DefaultOption> listIterator0 = (ListIterator<DefaultOption>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true).when(listIterator0).hasNext();
      doReturn((Object) null).when(listIterator0).next();
      doReturn((Object) null).when(listIterator0).previous();
      groupImpl0.process((WriteableCommandLine) null, listIterator0);
      assertEquals((-1), groupImpl0.getMaximum());
      assertEquals((-1), groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "7OLp\"XCvyugT|H", "7OLp\"XCvyugT|H", 91, 91, false);
      linkedList0.add((Object) groupImpl0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      writeableCommandLineImpl0.addSwitch(groupImpl0, false);
      // Undeclared exception!
      try { 
        groupImpl0.validate(writeableCommandLineImpl0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "7OLp\"XCvyugT|H", "7OLp\"XCvyugT|H", 91, 91, false);
      linkedList0.add((Object) groupImpl0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      writeableCommandLineImpl0.addSwitch(groupImpl0, false);
      GroupImpl groupImpl1 = new GroupImpl(linkedList0, "_8", "7OLp\"XCvyugT|H", 41, (-3504), false);
      try { 
        groupImpl1.validate(writeableCommandLineImpl0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unexpected 7OLp\"XCvyugT|H while processing _8
         //
         verifyException("org.apache.commons.cli2.option.GroupImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "DISPLAY_PARENT_ARGUMENT", "", 389, 2655, false);
      linkedList0.add(groupImpl0);
      GroupImpl groupImpl1 = new GroupImpl(linkedList0, "DISPLAY_PARENT_ARGUMENT", "DISPLAY_PARENT_ARGUMENT", 3, 2655, true);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl1, linkedList0);
      try { 
        groupImpl0.validate(writeableCommandLineImpl0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing option DISPLAY_PARENT_ARGUMENT
         //
         verifyException("org.apache.commons.cli2.option.GroupImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, "U;nptN/V6R[s0Z0mS", (-1811), (-2555), false);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      groupImpl0.validate(writeableCommandLineImpl0);
      assertEquals((-1811), groupImpl0.getMinimum());
      assertEquals((-2555), groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      LinkedList<Switch> linkedList0 = new LinkedList<Switch>();
      LinkedList<ArgumentImpl> linkedList1 = new LinkedList<ArgumentImpl>();
      DateValidator dateValidator0 = DateValidator.getDateInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("L!9", "L!9", 1, 1, 'G', 's', dateValidator0, "", linkedList0, (-534));
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0);
      linkedList1.add((ArgumentImpl) sourceDestArgument0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList1, (String) null, "U;nptN/V6R[s0Z0mS", (-1811), (-2555), false);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList1);
      try { 
        groupImpl0.validate(writeableCommandLineImpl0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing value(s) L!9
         //
         verifyException("org.apache.commons.cli2.option.ArgumentImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, 68, (-394), false);
      StringBuffer stringBuffer0 = new StringBuffer(".Lg)1+kI");
      LinkedHashSet<SourceDestArgument> linkedHashSet0 = new LinkedHashSet<SourceDestArgument>();
      Comparator<SourceDestArgument> comparator0 = (Comparator<SourceDestArgument>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      groupImpl0.appendUsage(stringBuffer0, (Set) linkedHashSet0, (Comparator) comparator0);
      assertEquals(68, groupImpl0.getMinimum());
      assertEquals(8, stringBuffer0.length());
      assertEquals((-394), groupImpl0.getMaximum());
      assertTrue(groupImpl0.isRequired());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      LinkedList<GroupImpl> linkedList1 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "X", 95, (-1), true);
      linkedList1.add(groupImpl0);
      GroupImpl groupImpl1 = new GroupImpl(linkedList1, "X", "", (-1), 95, true);
      linkedList1.offerLast(groupImpl0);
      String string0 = groupImpl1.toString();
      assertEquals(2, linkedList1.size());
      assertEquals("[X ( ()| ())]", string0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", 1043, (-251), false);
      LinkedList<Object> linkedList1 = new LinkedList<Object>();
      linkedList1.add((Object) groupImpl0);
      GroupImpl groupImpl1 = new GroupImpl(linkedList1, "", "", 1317, (-251), true);
      String string0 = groupImpl1.toString();
      assertEquals(1, linkedList1.size());
      assertEquals(" ([ ()])", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", (-1), (-1), false);
      LinkedHashSet<Object> linkedHashSet0 = new LinkedHashSet<Object>();
      List list0 = groupImpl0.helpLines((-1), linkedHashSet0, (Comparator) null);
      assertTrue(list0.isEmpty());
      assertEquals((-1), groupImpl0.getMaximum());
      assertEquals((-1), groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      PropertyOption propertyOption0 = new PropertyOption("org.apache.commons.cli2.option.GroupImpl", "DISPLAY_PARENT_ARGUMENT", 1441);
      linkedList0.add((Object) propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "DISPLAY_PARENT_ARGUMENT", "org.apache.commons.cli2.option.GroupImpl", 1441, 1441, false);
      groupImpl0.findOption("DISPLAY_PARENT_ARGUMENT");
      assertEquals(1, linkedList0.size());
      assertEquals(1441, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      PropertyOption propertyOption0 = new PropertyOption("org.apache.commons.cli2.option.GroupImpl", "DISPLAY_PARENT_ARGUMENT", 1441);
      linkedList0.add((Object) propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "DISPLAY_PARENT_ARGUMENT", "org.apache.commons.cli2.option.GroupImpl", 1441, 2, false);
      groupImpl0.findOption("org.apache.commons.cli2.option.GroupImpl");
      assertEquals(1, linkedList0.size());
      assertEquals(2, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "+QXF%T$i`MY3y9", "+QXF%T$i`MY3y9", 2655, 2655, false);
      linkedList0.add((Object) groupImpl0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      // Undeclared exception!
      try { 
        groupImpl0.defaults(writeableCommandLineImpl0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      NumberFormat numberFormat0 = NumberFormat.getCurrencyInstance();
      NumberValidator numberValidator0 = new NumberValidator(numberFormat0);
      ArgumentImpl argumentImpl0 = new ArgumentImpl("Sf", "Sf", 670, 670, 'D', 'D', numberValidator0, "Sf", linkedList0, 670);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0);
      linkedList0.add((Object) sourceDestArgument0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "KR(gz&81Gikss_V", "KR(gz&81Gikss_V", (-14), (-14), false);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      groupImpl0.defaults(writeableCommandLineImpl0);
      assertEquals(0, linkedList0.size());
      assertEquals((-14), groupImpl0.getMinimum());
  }
}
