/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:05:01 GMT 2023
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
import org.apache.commons.cli2.option.Command;
import org.apache.commons.cli2.option.DefaultOption;
import org.apache.commons.cli2.option.GroupImpl;
import org.apache.commons.cli2.option.PropertyOption;
import org.apache.commons.cli2.option.SourceDestArgument;
import org.apache.commons.cli2.validation.Validator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class GroupImpl_ESTest extends GroupImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Fzu:xl7eGwPNL", "", (-1215), (-1215), true);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      ListIterator<String> listIterator0 = (ListIterator<String>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true).when(listIterator0).hasNext();
      doReturn("Fzu:xl7eGwPNL").when(listIterator0).next();
      doReturn("[Fzu:xl7eGwPNL ()]").when(listIterator0).previous();
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertEquals("", groupImpl0.getDescription());
      assertEquals((-1215), groupImpl0.getMaximum());
      assertEquals((-1215), groupImpl0.getMinimum());
      assertEquals("Fzu:xl7eGwPNL", groupImpl0.getPreferredName());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      LinkedList<Command> linkedList0 = new LinkedList<Command>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "p5~ Ayw&E7-?#kVz", "p5~ Ayw&E7-?#kVz", 801, 801, false);
      groupImpl0.getAnonymous();
      assertEquals(801, groupImpl0.getMinimum());
      assertEquals(801, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      LinkedList<Command> linkedList0 = new LinkedList<Command>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, "", (-199), 0, false);
      int int0 = groupImpl0.getMaximum();
      assertEquals(0, int0);
      assertEquals((-199), groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("F-;", "", (-11), (-11), 'B', 'B', (Validator) null, "F-;", linkedList0, (-11));
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0, 'P', '\u0000', "F-;", linkedList0);
      linkedList0.offer(sourceDestArgument0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "F-;", "F-;", 793, 793, false);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "F-;");
      assertEquals(0, linkedList0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption("T>_;\\z]gS?Oxj", "", 0);
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "/{l%NDX4N*DS", "r/J7  *Jl", 1856, 1856, false);
      assertEquals(1, linkedList0.size());
      assertEquals("/{l%NDX4N*DS", groupImpl0.getPreferredName());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "F-;", "F-;", 793, 793, false);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "F-;");
      assertEquals(793, groupImpl0.getMaximum());
      assertEquals(793, groupImpl0.getMinimum());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Fzu:xl7eGwPNL", "Fzu:xl7eGwPNL", 793, 793, true);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, (String) null);
      assertFalse(boolean0);
      assertEquals(793, groupImpl0.getMaximum());
      assertEquals(793, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      LinkedList<Command> linkedList0 = new LinkedList<Command>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", (-2562), (-2562), false);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      LinkedList<ArgumentImpl> linkedList1 = new LinkedList<ArgumentImpl>();
      ListIterator<ArgumentImpl> listIterator0 = linkedList1.listIterator();
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertEquals((-2562), groupImpl0.getMinimum());
      assertEquals((-2562), groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      LinkedList<DefaultOption> linkedList0 = new LinkedList<DefaultOption>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "3d~,u_bq1v/@uTT1GA~", "3d~,u_bq1v/@uTT1GA~", 0, (-1), false);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      ListIterator<DefaultOption> listIterator0 = (ListIterator<DefaultOption>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true).when(listIterator0).hasNext();
      doReturn((DefaultOption) null).when(listIterator0).next();
      doReturn((DefaultOption) null).when(listIterator0).previous();
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertEquals(0, groupImpl0.getMinimum());
      assertEquals((-1), groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "F;h]:m", "F;h]:m", 61, 61, true);
      PropertyOption propertyOption0 = new PropertyOption("F;h]:m", "9,", 61);
      linkedList0.add(propertyOption0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      try { 
        groupImpl0.validate(writeableCommandLineImpl0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing option F;h]:m
         //
         verifyException("org.apache.commons.cli2.option.GroupImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "#e46CKat&WPw1);k3", "", 14, 14, false);
      linkedList0.addLast(groupImpl0);
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
  public void test11()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "#e46CKat&WPw1);k3", "", 14, 14, false);
      linkedList0.addLast(groupImpl0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      writeableCommandLineImpl0.addSwitch(groupImpl0, false);
      GroupImpl groupImpl1 = new GroupImpl(linkedList0, "", "", (-1), (-3534), false);
      try { 
        groupImpl1.validate(writeableCommandLineImpl0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unexpected #e46CKat&WPw1);k3 while processing 
         //
         verifyException("org.apache.commons.cli2.option.GroupImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", (-2421), (-3457), true);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      groupImpl0.validate(writeableCommandLineImpl0);
      assertEquals((-2421), groupImpl0.getMinimum());
      assertEquals((-3457), groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, Integer.MAX_VALUE, Integer.MAX_VALUE, true);
      LinkedHashSet<Command> linkedHashSet0 = new LinkedHashSet<Command>();
      Comparator<Object> comparator0 = (Comparator<Object>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      groupImpl0.appendUsage((StringBuffer) null, (Set) linkedHashSet0, (Comparator) comparator0, (String) null);
      assertTrue(groupImpl0.isRequired());
      assertEquals(Integer.MAX_VALUE, groupImpl0.getMinimum());
      assertEquals(Integer.MAX_VALUE, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", (-2421), (-3457), true);
      LinkedList<GroupImpl> linkedList1 = new LinkedList<GroupImpl>();
      linkedList1.addLast(groupImpl0);
      linkedList1.add(groupImpl0);
      GroupImpl groupImpl1 = new GroupImpl(linkedList1, "", "", (-2421), (-3457), true);
      String string0 = groupImpl1.toString();
      assertTrue(linkedList1.contains(groupImpl0));
      assertEquals("[ ([ ()]|[ ()])]", string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, Integer.MAX_VALUE, Integer.MAX_VALUE, true);
      LinkedHashSet<Command> linkedHashSet0 = new LinkedHashSet<Command>();
      Comparator<Command> comparator0 = (Comparator<Command>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      List list0 = groupImpl0.helpLines((-247), linkedHashSet0, comparator0);
      assertEquals(Integer.MAX_VALUE, groupImpl0.getMinimum());
      assertEquals(0, list0.size());
      assertEquals(Integer.MAX_VALUE, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      LinkedList<Command> linkedList0 = new LinkedList<Command>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "5wgqrM$0>EeCOcH", 0, (-3534), true);
      LinkedList<GroupImpl> linkedList1 = new LinkedList<GroupImpl>();
      linkedList1.add(groupImpl0);
      GroupImpl groupImpl1 = new GroupImpl(linkedList1, "", "", (-3534), 0, false);
      groupImpl1.findOption("5wgqrM$0>EeCOcH");
      assertTrue(linkedList1.contains(groupImpl0));
      assertEquals((-3534), groupImpl1.getMinimum());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Fzu:xl7eGwPNL", "", (-1215), (-1215), true);
      groupImpl0.defaults((WriteableCommandLine) null);
      assertEquals((-1215), groupImpl0.getMinimum());
      assertEquals("", groupImpl0.getDescription());
      assertEquals("Fzu:xl7eGwPNL", groupImpl0.getPreferredName());
      assertEquals((-1215), groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Fzu:xl7eGwPNL", "Fzu:xl7eGwPNL", 796, 796, true);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      linkedList0.add(groupImpl0);
      // Undeclared exception!
      try { 
        groupImpl0.defaults(writeableCommandLineImpl0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
      }
  }
}