/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 12:59:51 GMT 2023
 */

package org.apache.commons.cli2.option;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.text.DecimalFormat;
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
import org.apache.commons.cli2.option.GroupImpl;
import org.apache.commons.cli2.option.PropertyOption;
import org.apache.commons.cli2.option.SourceDestArgument;
import org.apache.commons.cli2.option.Switch;
import org.apache.commons.cli2.validation.NumberValidator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class GroupImpl_ESTest extends GroupImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "]3C'", "&x};PkE2Y{+C}+9|Q", (-805), (-1));
      groupImpl0.getAnonymous();
      assertEquals("]3C'", groupImpl0.getPreferredName());
      assertEquals((-1), groupImpl0.getMaximum());
      assertEquals("&x};PkE2Y{+C}+9|Q", groupImpl0.getDescription());
      assertEquals((-805), groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "YqA(;<G']", "YqA(;<G']", 811, 811);
      LinkedList<GroupImpl> linkedList1 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl1 = new GroupImpl(linkedList1, "YqA(;<G']", "YqA(;<G']", 811, 811);
      linkedList1.add(groupImpl0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      try { 
        groupImpl1.validate(writeableCommandLineImpl0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing option YqA(;<G']
         //
         verifyException("org.apache.commons.cli2.option.GroupImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, ".jZJ", 0, (-392));
      int int0 = groupImpl0.getMaximum();
      assertEquals((-392), int0);
      assertEquals(0, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "YqA(;<G']", "YqA(;<G']", 0, 0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      writeableCommandLineImpl0.addValue(groupImpl0, linkedList0);
      assertEquals(0, groupImpl0.getMinimum());
      assertEquals(0, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      DecimalFormat decimalFormat0 = new DecimalFormat();
      NumberValidator numberValidator0 = new NumberValidator(decimalFormat0);
      ArgumentImpl argumentImpl0 = new ArgumentImpl("lW", "lW", 0, 0, '~', '%', numberValidator0, "lW", linkedList0, 0);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0);
      linkedList0.add((ArgumentImpl) sourceDestArgument0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, ")dQcbi", "", 1644, (-2964));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      groupImpl0.defaults(writeableCommandLineImpl0);
      assertEquals(0, linkedList0.size());
      assertTrue(groupImpl0.isRequired());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      LinkedList<Switch> linkedList0 = new LinkedList<Switch>();
      LinkedHashSet<Object> linkedHashSet0 = new LinkedHashSet<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "9X=F?%\"s", "9X=F?%\"s", (-2280), 1445);
      Boolean boolean0 = Boolean.TRUE;
      Switch switch0 = new Switch("org.apache.commons.cli2.option.DefaultOption", "9X=F?%\"s", "org.apache.commons.cli2.option.DefaultOption", linkedHashSet0, "org.apache.commons.cli2.option.DefaultOption", true, (Argument) null, groupImpl0, (-1), boolean0);
      linkedList0.add(switch0);
      GroupImpl groupImpl1 = new GroupImpl(linkedList0, (String) null, "I&jYi_y@J+#y", (-2280), (-2280));
      assertTrue(linkedList0.contains(switch0));
      assertFalse(groupImpl1.isRequired());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", (-2673), (-2673));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, (String) null);
      assertFalse(boolean0);
      assertEquals((-2673), groupImpl0.getMinimum());
      assertEquals((-2673), groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", (-2673), (-2673));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "");
      assertEquals((-2673), groupImpl0.getMinimum());
      assertFalse(boolean0);
      assertEquals((-2673), groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "YqA(;<G']", "YqA(;<G']", 0, 0);
      LinkedList<SourceDestArgument> linkedList1 = new LinkedList<SourceDestArgument>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList1);
      LinkedList<Object> linkedList2 = new LinkedList<Object>();
      ListIterator<Object> listIterator0 = linkedList2.listIterator();
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertEquals(0, groupImpl0.getMaximum());
      assertEquals(0, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "4xHR`i[j:6kG:", (-1238), 91);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      ListIterator<GroupImpl> listIterator0 = (ListIterator<GroupImpl>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true).when(listIterator0).hasNext();
      doReturn((Object) null).when(listIterator0).next();
      doReturn((Object) null).when(listIterator0).previous();
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertEquals((-1238), groupImpl0.getMinimum());
      assertEquals(91, groupImpl0.getMaximum());
      assertEquals("", groupImpl0.getPreferredName());
      assertEquals("4xHR`i[j:6kG:", groupImpl0.getDescription());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "~%-5$flB", "~%-5$flB", 1736, 1736);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      PropertyOption propertyOption0 = new PropertyOption("|", "|", 1736);
      linkedList0.add(propertyOption0);
      try { 
        groupImpl0.validate(writeableCommandLineImpl0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing option ~%-5$flB
         //
         verifyException("org.apache.commons.cli2.option.GroupImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "YqA(;<G']", "YqA(;<G']", 0, 0);
      String string0 = groupImpl0.toString();
      assertEquals("[YqA(;<G'] ()]", string0);
      assertEquals(0, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      LinkedList<SourceDestArgument> linkedList0 = new LinkedList<SourceDestArgument>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, "org.apache.commons.cli2.option.DefaultOption", 93, 93);
      StringBuffer stringBuffer0 = new StringBuffer(93);
      LinkedHashSet<ArgumentImpl> linkedHashSet0 = new LinkedHashSet<ArgumentImpl>();
      Comparator<GroupImpl> comparator0 = (Comparator<GroupImpl>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      groupImpl0.appendUsage(stringBuffer0, (Set) linkedHashSet0, (Comparator) comparator0, (String) null);
      assertEquals(93, groupImpl0.getMaximum());
      assertEquals(0, stringBuffer0.length());
      assertEquals(93, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, ".jZJ", 0, (-392));
      LinkedList<GroupImpl> linkedList1 = new LinkedList<GroupImpl>();
      linkedList1.offerLast(groupImpl0);
      GroupImpl groupImpl1 = new GroupImpl(linkedList1, (String) null, "", 811, (-506));
      linkedList1.add(groupImpl0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl1, linkedList1);
      try { 
        groupImpl1.validate(writeableCommandLineImpl0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing option |
         //
         verifyException("org.apache.commons.cli2.option.GroupImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      LinkedHashSet<Command> linkedHashSet0 = new LinkedHashSet<Command>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "4xHR`i[j:6kG:", (-1238), 91);
      Comparator<String> comparator0 = (Comparator<String>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      List list0 = groupImpl0.helpLines((-1238), linkedHashSet0, comparator0);
      assertEquals(0, list0.size());
      assertEquals("", groupImpl0.getPreferredName());
      assertEquals("4xHR`i[j:6kG:", groupImpl0.getDescription());
      assertEquals(91, groupImpl0.getMaximum());
      assertEquals((-1238), groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, ".jZJ", 0, (-392));
      LinkedList<GroupImpl> linkedList1 = new LinkedList<GroupImpl>();
      linkedList1.offerLast(groupImpl0);
      GroupImpl groupImpl1 = new GroupImpl(linkedList1, (String) null, "", 811, (-506));
      groupImpl1.findOption((String) null);
      assertTrue(linkedList1.contains(groupImpl0));
      assertEquals(811, groupImpl1.getMinimum());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      LinkedList<SourceDestArgument> linkedList0 = new LinkedList<SourceDestArgument>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, "org.apache.commons.cli2.option.DefaultOption", 93, 93);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      groupImpl0.defaults(writeableCommandLineImpl0);
      assertEquals(93, groupImpl0.getMinimum());
      assertEquals(93, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      LinkedList<String> linkedList0 = new LinkedList<String>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", (-759), (-759));
      linkedList0.add("");
      // Undeclared exception!
      try { 
        groupImpl0.defaults((WriteableCommandLine) null);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // java.lang.String cannot be cast to org.apache.commons.cli2.Option
         //
         verifyException("org.apache.commons.cli2.option.GroupImpl", e);
      }
  }
}