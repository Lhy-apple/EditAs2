/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:03:50 GMT 2023
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
import org.apache.commons.cli2.Argument;
import org.apache.commons.cli2.WriteableCommandLine;
import org.apache.commons.cli2.commandline.WriteableCommandLineImpl;
import org.apache.commons.cli2.option.ArgumentImpl;
import org.apache.commons.cli2.option.Command;
import org.apache.commons.cli2.option.DefaultOption;
import org.apache.commons.cli2.option.GroupImpl;
import org.apache.commons.cli2.option.PropertyOption;
import org.apache.commons.cli2.option.SourceDestArgument;
import org.apache.commons.cli2.validation.NumberValidator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class GroupImpl_ESTest extends GroupImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      LinkedList<Command> linkedList0 = new LinkedList<Command>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "property", 0, 0);
      String string0 = groupImpl0.getPreferredName();
      assertEquals(0, groupImpl0.getMinimum());
      assertEquals("property", groupImpl0.getDescription());
      assertEquals("", string0);
      assertEquals(0, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "j5 =U@n>/~YKq5#_", "j5 =U@n>/~YKq5#_", 0, 0);
      groupImpl0.getAnonymous();
      assertEquals(0, groupImpl0.getMinimum());
      assertEquals(0, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "V;SdEyoh", "A-7;&}z^+|RlO", 0, 91);
      int int0 = groupImpl0.getMaximum();
      assertEquals(91, int0);
      assertEquals("A-7;&}z^+|RlO", groupImpl0.getDescription());
      assertEquals("V;SdEyoh", groupImpl0.getPreferredName());
      assertEquals(0, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "V;SdEyoh", "A-7;&}z^+|RlO", 0, 91);
      String string0 = groupImpl0.getDescription();
      assertEquals("V;SdEyoh", groupImpl0.getPreferredName());
      assertEquals(91, groupImpl0.getMaximum());
      assertEquals("A-7;&}z^+|RlO", string0);
      assertEquals(0, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      NumberValidator numberValidator0 = NumberValidator.getCurrencyInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("4-X<X^_yS4Z'WtC%uF7", "Enum.illegal.value", 0, 0, 'Y', '3', numberValidator0, "Enum.illegal.value", linkedList0, Integer.MAX_VALUE);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0, 'Y', '3', "e*&!Ws|O", linkedList0);
      linkedList0.add((ArgumentImpl) sourceDestArgument0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "j5 =U@n>/~YKq5#_", "j5 =U@n>/~YKq5#_", 0, 0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      groupImpl0.validate(writeableCommandLineImpl0);
      assertEquals(0, linkedList0.size());
      assertEquals(0, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "R^hw9.~x", "j.*40<;", 0, (-701));
      LinkedList<GroupImpl> linkedList1 = new LinkedList<GroupImpl>();
      linkedList1.add(groupImpl0);
      GroupImpl groupImpl1 = new GroupImpl(linkedList1, "j5 =U@n>/~YKq5#_", "R^hw9.~x", 0, (-1358));
      assertTrue(linkedList1.contains(groupImpl0));
      assertEquals((-1358), groupImpl1.getMaximum());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      LinkedList<DefaultOption> linkedList0 = new LinkedList<DefaultOption>();
      LinkedHashSet<Integer> linkedHashSet0 = new LinkedHashSet<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "JiU", "%8IjXrfsKp z[rI[U", (-1), 184);
      DefaultOption defaultOption0 = new DefaultOption("JiU", "JiU", true, "JiU", "JiU", linkedHashSet0, linkedHashSet0, true, (Argument) null, groupImpl0, 1354);
      linkedList0.addLast(defaultOption0);
      GroupImpl groupImpl1 = new GroupImpl(linkedList0, "--", "--", (-1), (-1));
      assertTrue(linkedList0.contains(defaultOption0));
      assertEquals((-1), groupImpl1.getMaximum());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      NumberValidator numberValidator0 = NumberValidator.getPercentInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("ClassValidator.class.notfound", "ClassValidator.class.notfound", 0, 0, 'Z', 'Z', numberValidator0, ",Y2O8rOr", linkedList0, 0);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0, 'z', 'Z', "", linkedList0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "--", "--", 0, 0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(sourceDestArgument0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "ClassValidator.class.notfound");
      assertEquals(0, groupImpl0.getMaximum());
      assertFalse(boolean0);
      assertEquals(0, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Oz<ceUI}/O<|u|%bF", "Oz<ceUI}/O<|u|%bF", (-3095), 93);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, (String) null);
      assertEquals((-3095), groupImpl0.getMinimum());
      assertFalse(boolean0);
      assertEquals(93, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      PropertyOption propertyOption0 = new PropertyOption("", "", 0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "|=8$URdF'S9gw", "", 19, 19);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "");
      assertEquals(19, groupImpl0.getMaximum());
      assertEquals("", groupImpl0.getDescription());
      assertEquals(19, groupImpl0.getMinimum());
      assertEquals("|=8$URdF'S9gw", groupImpl0.getPreferredName());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      NumberValidator numberValidator0 = NumberValidator.getPercentInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("ClassValidator.class.notfound", "ClassValidator.class.notfound", 0, 0, 'Z', 'Z', numberValidator0, ",Y2O8rOr", linkedList0, 0);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0, 'z', 'Z', "", linkedList0);
      linkedList0.add((ArgumentImpl) sourceDestArgument0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "--", "--", 0, 0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(sourceDestArgument0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "ClassValidator.class.notfound");
      assertEquals(0, linkedList0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      NumberValidator numberValidator0 = NumberValidator.getPercentInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("ClassValidator.class.notfound", "ClassValidator.class.notfound", 0, 0, 'G', 'G', numberValidator0, "nEX?>wnZ^][b", linkedList0, 0);
      LinkedList linkedList1 = (LinkedList)linkedList0.clone();
      ListIterator<String> listIterator0 = (ListIterator<String>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true).when(listIterator0).hasNext();
      doReturn("").when(listIterator0).next();
      doReturn("m~Pjo`*o9@%vSJt2").when(listIterator0).previous();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(argumentImpl0, linkedList0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList1, "ClassValidator.class.notfound", "j5 =U@n>/~YKq5#_", 0, 68);
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertEquals(68, groupImpl0.getMaximum());
      assertEquals(0, groupImpl0.getMinimum());
      assertEquals("ClassValidator.class.notfound", groupImpl0.getPreferredName());
      assertEquals("j5 =U@n>/~YKq5#_", groupImpl0.getDescription());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      LinkedHashSet<Object> linkedHashSet0 = new LinkedHashSet<Object>();
      NumberValidator numberValidator0 = NumberValidator.getPercentInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("ClassValidator.class.notfound", "ClassValidator.class.notfound", 0, 0, 'Z', 'Z', numberValidator0, ",Y2O8rOr", linkedList0, 0);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0, 'n', 'Z', "", linkedList0);
      linkedList0.add((ArgumentImpl) sourceDestArgument0);
      Integer integer0 = new Integer(0);
      Comparator<Object> comparator0 = (Comparator<Object>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      LinkedList linkedList1 = (LinkedList)linkedList0.clone();
      LinkedList<Object> linkedList2 = new LinkedList<Object>();
      ListIterator<String> listIterator0 = (ListIterator<String>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true).when(listIterator0).hasNext();
      doReturn((Object) null).when(listIterator0).next();
      doReturn((Object) null).when(listIterator0).previous();
      GroupImpl groupImpl0 = new GroupImpl(linkedList1, "--", "ClassValidator.class.notfound", (-4833), 0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(sourceDestArgument0, linkedList0);
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertEquals(0, linkedList1.size());
      assertEquals((-4833), groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      NumberValidator numberValidator0 = NumberValidator.getPercentInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("ClassValidator.class.notfound", "ClassValidator.class.notfound", 0, 0, 'G', 'G', numberValidator0, "nEX?>wnZ^][b", linkedList0, 0);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0, 'r', 'G', "", linkedList0);
      linkedList0.add((ArgumentImpl) sourceDestArgument0);
      LinkedList linkedList1 = (LinkedList)linkedList0.clone();
      ListIterator<String> listIterator0 = (ListIterator<String>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true, false, false).when(listIterator0).hasNext();
      doReturn("").when(listIterator0).next();
      doReturn("m~Pjo`*o9@%vSJt2").when(listIterator0).previous();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(argumentImpl0, linkedList0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList1, "ClassValidator.class.notfound", "j5 =U@n>/~YKq5#_", 0, 68);
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertEquals(0, linkedList1.size());
      assertEquals("ClassValidator.class.notfound", groupImpl0.getPreferredName());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      NumberValidator numberValidator0 = NumberValidator.getCurrencyInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("4-X<X^_yS4Z'WtC%uF7", "Enum.illegal.value", 0, 0, 'Y', '3', numberValidator0, "Enum.illegal.value", linkedList0, Integer.MAX_VALUE);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0, 'Y', '3', "e*&!Ws|O", linkedList0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "j5 =U@n>/~YKq5#_", "j5 =U@n>/~YKq5#_", 0, 0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      linkedList0.add((ArgumentImpl) sourceDestArgument0);
      groupImpl0.validate(writeableCommandLineImpl0);
      assertEquals(0, groupImpl0.getMinimum());
      assertEquals(0, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "true", (String) null, 2937, 2937);
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
  public void test16()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Hlo\"AmK:@j,+jd", "Hlo\"AmK:@j,+jd", (-701), 0);
      linkedList0.add(groupImpl0);
      LinkedHashSet<Integer> linkedHashSet0 = new LinkedHashSet<Integer>();
      List list0 = groupImpl0.helpLines(34, linkedHashSet0, (Comparator) null);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, list0);
      // Undeclared exception!
      try { 
        groupImpl0.validate(writeableCommandLineImpl0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "fA.rPG,0B]X5L^d'V", "fA.rPG,0B]X5L^d'V", 93, 93);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      LinkedList<PropertyOption> linkedList1 = new LinkedList<PropertyOption>();
      NumberValidator numberValidator0 = NumberValidator.getPercentInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("+", "ClassValidator.class.notfound", 0, 0, 'u', 'F', numberValidator0, "+", linkedList1, (-1));
      linkedList0.add(argumentImpl0);
      writeableCommandLineImpl0.addSwitch(argumentImpl0, true);
      try { 
        groupImpl0.validate(writeableCommandLineImpl0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing option fA.rPG,0B]X5L^d'V
         //
         verifyException("org.apache.commons.cli2.option.GroupImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "j5 =U@n>/~YKq5#_", "j5 =U@n>/~YKq5#_", 0, 0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      NumberValidator numberValidator0 = NumberValidator.getPercentInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("ClassValidator.class.notfound", "ClassValidator.class.notfound", 0, 0, 'G', 'G', numberValidator0, "nEX?>wnZ^][b", linkedList0, 0);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0, 'r', 'G', "--", linkedList0);
      linkedList0.add((ArgumentImpl) sourceDestArgument0);
      writeableCommandLineImpl0.addSwitch(sourceDestArgument0, true);
      try { 
        groupImpl0.validate(writeableCommandLineImpl0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unexpected SourceDestArgument while processing j5 =U@n>/~YKq5#_
         //
         verifyException("org.apache.commons.cli2.option.GroupImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, (-921), (-104));
      Set set0 = groupImpl0.getPrefixes();
      StringBuffer stringBuffer0 = new StringBuffer((CharSequence) "17");
      Comparator<Object> comparator0 = (Comparator<Object>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      groupImpl0.appendUsage(stringBuffer0, set0, comparator0);
      assertEquals((-104), groupImpl0.getMaximum());
      assertEquals((-921), groupImpl0.getMinimum());
      assertEquals("17", stringBuffer0.toString());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "j5 =U@n>/~YKq5#_", "j5 =U@n>/~YKq5#_", 0, 0);
      NumberValidator numberValidator0 = NumberValidator.getPercentInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("ClassValidator.class.notfound", "ClassValidator.class.notfound", 0, 0, 'G', 'G', numberValidator0, "nEX?>wnZ^][b", linkedList0, 0);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0, 'r', 'G', "", linkedList0);
      linkedList0.add((ArgumentImpl) sourceDestArgument0);
      linkedList0.add((ArgumentImpl) sourceDestArgument0);
      String string0 = groupImpl0.toString();
      assertEquals("[j5 =U@n>/~YKq5#_ (|)]", string0);
      assertEquals(0, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      NumberValidator numberValidator0 = NumberValidator.getCurrencyInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("4-X<X^_yS4Z'WtC%uF7", "Enum.illegal.value", 0, 0, 'Y', '3', numberValidator0, "Enum.illegal.value", linkedList0, Integer.MAX_VALUE);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0, 'Y', '3', "e*&!Ws|O", linkedList0);
      linkedList0.add((ArgumentImpl) sourceDestArgument0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "j5 =U@n>/~YKq5#_", "j5 =U@n>/~YKq5#_", 0, 0);
      String string0 = groupImpl0.toString();
      assertEquals(0, linkedList0.size());
      assertEquals("[j5 =U@n>/~YKq5#_ ()] ", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      NumberValidator numberValidator0 = NumberValidator.getCurrencyInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("4-X<X^_yS4Z'WtC%uF7", "Enum.illegal.value", 0, 0, 'Y', '3', numberValidator0, "Enum.illegal.value", linkedList0, Integer.MAX_VALUE);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0, 'Y', '3', "e*&!Ws|O", linkedList0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "j5 =U@n>/~YKq5#_", "j5 =U@n>/~YKq5#_", 0, 0);
      linkedList0.add((ArgumentImpl) sourceDestArgument0);
      groupImpl0.findOption("4-X<X^_yS4Z'WtC%uF7");
      assertEquals(0, groupImpl0.getMaximum());
      assertEquals(0, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "j5 =U@n>/~YKq5#_", "j5 =U@n>/~YKq5#_", 0, 0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      NumberValidator numberValidator0 = NumberValidator.getIntegerInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("ClassValidator.class.notfound", "ClassValidator.class.notfound", 0, 0, 'U', 'U', numberValidator0, "nEX?>wnZ^][b", linkedList0, 0);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0, 'I', 'U', "", linkedList0);
      linkedList0.add((ArgumentImpl) sourceDestArgument0);
      groupImpl0.defaults(writeableCommandLineImpl0);
      assertEquals(0, groupImpl0.getMinimum());
      assertEquals(0, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      NumberValidator numberValidator0 = NumberValidator.getCurrencyInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("4-X<X^_yS4Z'WtC%uF7", "Enum.illegal.value", 0, 0, 'Y', '3', numberValidator0, "Enum.illegal.value", linkedList0, Integer.MAX_VALUE);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0, 'Y', '3', "e*&!Ws|O", linkedList0);
      linkedList0.add((ArgumentImpl) sourceDestArgument0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "j5 =U@n>/~YKq5#_", "j5 =U@n>/~YKq5#_", 0, 0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      groupImpl0.defaults(writeableCommandLineImpl0);
      assertEquals(0, linkedList0.size());
      assertEquals(0, groupImpl0.getMaximum());
  }
}