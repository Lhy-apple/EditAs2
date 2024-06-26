/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:49:34 GMT 2023
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
import java.util.Set;
import org.apache.commons.cli2.WriteableCommandLine;
import org.apache.commons.cli2.commandline.WriteableCommandLineImpl;
import org.apache.commons.cli2.option.ArgumentImpl;
import org.apache.commons.cli2.option.DefaultOption;
import org.apache.commons.cli2.option.GroupImpl;
import org.apache.commons.cli2.option.PropertyOption;
import org.apache.commons.cli2.option.SourceDestArgument;
import org.apache.commons.cli2.validation.DateValidator;
import org.apache.commons.cli2.validation.EnumValidator;
import org.apache.commons.cli2.validation.NumberValidator;
import org.apache.commons.cli2.validation.Validator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class GroupImpl_ESTest extends GroupImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "R_UJjG", "R_UJjG", (-2794), (-2794), true);
      groupImpl0.equals(groupImpl0);
      assertEquals((-2794), groupImpl0.getMinimum());
      assertEquals((-2794), groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "Passes properties and values to the application", 0, 0, true);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      ListIterator<String> listIterator0 = (ListIterator<String>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true).when(listIterator0).hasNext();
      doReturn("-D").when(listIterator0).next();
      doReturn("Passes properties and values to the application").when(listIterator0).previous();
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertEquals(0, groupImpl0.getMaximum());
      assertEquals(0, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "R_UJjG", "R_UJjG", 0, (-1), false);
      groupImpl0.getAnonymous();
      assertEquals((-1), groupImpl0.getMaximum());
      assertEquals(0, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", (String) null, (-2794), 0, true);
      int int0 = groupImpl0.getMaximum();
      assertEquals((-2794), groupImpl0.getMinimum());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      LinkedHashSet<GroupImpl> linkedHashSet0 = new LinkedHashSet<GroupImpl>();
      EnumValidator enumValidator0 = new EnumValidator(linkedHashSet0);
      ArgumentImpl argumentImpl0 = new ArgumentImpl("Z)3zEfdlc;)~e[4a!{?", "Missing.option", (-2299), (-2299), 'z', '#', enumValidator0, "w>zh;n+&QayFQDFce", linkedList0, (-2299));
      linkedList0.add(argumentImpl0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, (-2794), 53, false);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      groupImpl0.defaults(writeableCommandLineImpl0);
      assertFalse(linkedList0.contains(argumentImpl0));
      assertEquals((-2794), groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, (-2794), (-2794), false);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, (String) null);
      assertFalse(boolean0);
      assertEquals((-2794), groupImpl0.getMinimum());
      assertEquals((-2794), groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "Passes properties and values to the application", 0, 0, true);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "-D");
      assertEquals(1, linkedList0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption("", "", 93);
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "-D", "$L", (-1198), (-1198), true);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "Passes properties and values to the application");
      assertEquals(1, linkedList0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "Passes properties and values to the application", 0, 0, true);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "Passes properties and values to the application");
      assertEquals(1, linkedList0.size());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "-D", "Passes properties and values to the application", 68, 68, true);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "-D");
      assertEquals(68, groupImpl0.getMaximum());
      assertEquals("-D", groupImpl0.getPreferredName());
      assertEquals(68, groupImpl0.getMinimum());
      assertEquals("Passes properties and values to the application", groupImpl0.getDescription());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      NumberValidator numberValidator0 = NumberValidator.getNumberInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("P8$8[p_.+HP/B{_3#", "", (-1156), (-1156), 'z', 'R', numberValidator0, "{J'a9\"", linkedList0, (-111));
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0, 'm', 'q', "{J'a9\"", linkedList0);
      linkedList0.addLast(sourceDestArgument0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "{J'a9\"", "", (-2357), (-2357), false);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "");
      assertEquals(0, linkedList0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, (-2794), 53, false);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      ListIterator<ArgumentImpl> listIterator0 = (ListIterator<ArgumentImpl>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(listIterator0).hasNext();
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertEquals(53, groupImpl0.getMaximum());
      assertEquals((-2794), groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "Passes properties and values to the application", 0, 0, true);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      ListIterator<String> listIterator0 = (ListIterator<String>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true).when(listIterator0).hasNext();
      doReturn("-D", (Object) null).when(listIterator0).next();
      doReturn("Passes properties and values to the application").when(listIterator0).previous();
      try { 
        groupImpl0.process(writeableCommandLineImpl0, listIterator0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unexpected -D while processing 
         //
         verifyException("org.apache.commons.cli2.option.PropertyOption", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "Passes properties and values to the application", 0, 0, true);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      groupImpl0.validate(writeableCommandLineImpl0);
      assertEquals(1, linkedList0.size());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "Passes properties and values to the application", (-630), (-8), true);
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
  public void test15()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "Passes properties and values to the application", (-630), 1, true);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      writeableCommandLineImpl0.addOption(propertyOption0);
      groupImpl0.validate(writeableCommandLineImpl0);
      assertEquals(1, linkedList0.size());
      assertEquals((-630), groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "Passes properties and values to the application", 919, 919, true);
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
  public void test17()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      LinkedList<Object> linkedList1 = new LinkedList<Object>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("DISPLAY_OPTIONAL_CHILD_GROUP", "DISPLAY_OPTIONAL_CHILD_GROUP", 0, 32, 'i', 'L', (Validator) null, "DISPLAY_OPTIONAL_CHILD_GROUP", linkedList1, 44);
      linkedList0.add(argumentImpl0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "DISPLAY_OPTIONAL_CHILD_GROUP", "DISPLAY_OPTIONAL_CHILD_GROUP", 0, 0, true);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      groupImpl0.validate(writeableCommandLineImpl0);
      assertFalse(linkedList0.contains(argumentImpl0));
      assertEquals(0, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "org.apache.commons.cli2.option.DefaultOption", "org.apache.commons.cli2.option.DefaultOption", (-2685), (-2685), false);
      linkedList0.add(groupImpl0);
      // Undeclared exception!
      groupImpl0.toString();
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, (-2794), (-2794), false);
      LinkedHashSet<Integer> linkedHashSet0 = new LinkedHashSet<Integer>();
      Comparator<Object> comparator0 = (Comparator<Object>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      groupImpl0.appendUsage((StringBuffer) null, (Set) linkedHashSet0, (Comparator) comparator0, "org.apache.commons.cli2.option.PropertyOption");
      assertEquals((-2794), groupImpl0.getMaximum());
      assertEquals((-2794), groupImpl0.getMinimum());
      assertFalse(groupImpl0.isRequired());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add(propertyOption0);
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", (-2789), (-2789), false);
      String string0 = groupImpl0.toString();
      assertEquals(2, linkedList0.size());
      assertEquals("[ (-D<property>=<value>|-D<property>=<value>)]", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      LinkedHashSet<DefaultOption> linkedHashSet0 = new LinkedHashSet<DefaultOption>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "Passes properties and values to the application", 0, 0, true);
      Comparator<Object> comparator0 = (Comparator<Object>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      List list0 = groupImpl0.helpLines(0, linkedHashSet0, comparator0);
      assertTrue(list0.isEmpty());
      assertEquals(0, groupImpl0.getMaximum());
      assertEquals(0, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", (-2789), (-2789), false);
      groupImpl0.findOption("-D");
      assertEquals(1, linkedList0.size());
      assertEquals((-2789), groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "Passes properties and values to the application", 0, 0, true);
      groupImpl0.findOption("Unexpected.token");
      assertEquals(1, linkedList0.size());
      assertEquals(0, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "4", "4", (-1543), (-1543), false);
      DateValidator dateValidator0 = new DateValidator();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("p![&F`k", "W@?", (-1543), (-1290), '3', 'C', dateValidator0, "", linkedList0, 1735);
      groupImpl0.setParent(argumentImpl0);
      String string0 = groupImpl0.toString();
      assertEquals((-1543), groupImpl0.getMinimum());
      assertEquals((-1543), groupImpl0.getMaximum());
      assertEquals("[4 ()]", string0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "R_UJjG", "R_UJjG", (-2794), (-2794), true);
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      groupImpl0.setParent(propertyOption0);
      String string0 = groupImpl0.toString();
      assertEquals("[R_UJjG ()]", string0);
      assertEquals((-2794), groupImpl0.getMinimum());
      assertEquals((-2794), groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "Passes properties and values to the application", 0, 0, true);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      groupImpl0.defaults(writeableCommandLineImpl0);
      assertEquals(1, linkedList0.size());
  }
}
