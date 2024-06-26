/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 17:44:50 GMT 2023
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
import org.apache.commons.cli2.DisplaySetting;
import org.apache.commons.cli2.WriteableCommandLine;
import org.apache.commons.cli2.commandline.WriteableCommandLineImpl;
import org.apache.commons.cli2.option.ArgumentImpl;
import org.apache.commons.cli2.option.Command;
import org.apache.commons.cli2.option.GroupImpl;
import org.apache.commons.cli2.option.PropertyOption;
import org.apache.commons.cli2.option.SourceDestArgument;
import org.apache.commons.cli2.validation.FileValidator;
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
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Yf6(;H<G4]", "Yf6(;H<G4]", 0, 0);
      String string0 = groupImpl0.toString();
      assertEquals(0, groupImpl0.getMaximum());
      assertEquals("[Yf6(;H<G4] ()]", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "]3C'", "&x};PkE2Y{+C}+9|Q", (-805), (-1));
      groupImpl0.getAnonymous();
      assertEquals("&x};PkE2Y{+C}+9|Q", groupImpl0.getDescription());
      assertEquals((-805), groupImpl0.getMinimum());
      assertEquals((-1), groupImpl0.getMaximum());
      assertEquals("]3C'", groupImpl0.getPreferredName());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, "YqA(;<G']", 0, 811);
      int int0 = groupImpl0.getMaximum();
      assertEquals(811, int0);
      assertEquals(0, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      DecimalFormat decimalFormat0 = new DecimalFormat();
      NumberValidator numberValidator0 = new NumberValidator(decimalFormat0);
      ArgumentImpl argumentImpl0 = new ArgumentImpl("lW", "lW", 0, 0, '~', '%', numberValidator0, "lW", linkedList0, 0);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0);
      linkedList0.add((ArgumentImpl) sourceDestArgument0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, ")dQcbi", "", '\u0000', (-2964));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      groupImpl0.defaults(writeableCommandLineImpl0);
      assertEquals(0, linkedList0.size());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      LinkedList<Command> linkedList0 = new LinkedList<Command>();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      FileValidator fileValidator0 = FileValidator.getExistingDirectoryInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("l$S", "tNL@Q^_C\"d?\"OfI??*<", 0, 1855, '\'', 'D', fileValidator0, "", linkedList0, 1855);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "l$S", 1855, 1131);
      Command command0 = new Command("org.apache.commons.cli2.option.Command", "l$S", linkedHashSet0, false, argumentImpl0, groupImpl0, 0);
      linkedList0.add(command0);
      GroupImpl groupImpl1 = new GroupImpl(linkedList0, "tNL@Q^_C\"d?\"OfI??*<", "tNL@Q^_C\"d?\"OfI??*<", 0, 798);
      assertTrue(linkedList0.contains(command0));
      assertEquals(1, linkedList0.size());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "@gw`a)VKGDkKTI2", "YqA(;<G']", 68, (-2673));
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "-D");
      assertFalse(boolean0);
      assertEquals((-2673), groupImpl0.getMaximum());
      assertEquals("@gw`a)VKGDkKTI2", groupImpl0.getPreferredName());
      assertEquals("YqA(;<G']", groupImpl0.getDescription());
      assertEquals(68, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", (-2673), (-2673));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, (String) null);
      assertEquals((-2673), groupImpl0.getMaximum());
      assertEquals((-2673), groupImpl0.getMinimum());
      assertFalse(boolean0);
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
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Yf6(;H<G4]", "Yf6(;H<G4]", 0, 0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      LinkedList<ArgumentImpl> linkedList1 = new LinkedList<ArgumentImpl>();
      ListIterator<ArgumentImpl> listIterator0 = linkedList1.listIterator();
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertEquals(0, groupImpl0.getMinimum());
      assertEquals(0, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "4xHR`i[j:6kG:", (-1238), 91);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      ListIterator<GroupImpl> listIterator0 = (ListIterator<GroupImpl>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true).when(listIterator0).hasNext();
      doReturn((GroupImpl) null).when(listIterator0).next();
      doReturn((GroupImpl) null).when(listIterator0).previous();
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertEquals("4xHR`i[j:6kG:", groupImpl0.getDescription());
      assertEquals((-1238), groupImpl0.getMinimum());
      assertEquals(91, groupImpl0.getMaximum());
      assertEquals("", groupImpl0.getPreferredName());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "aL{kjXZnn9", "aL{kjXZnn9", 4280, 0);
      linkedList0.offerLast(groupImpl0);
      LinkedList<DisplaySetting> linkedList1 = new LinkedList<DisplaySetting>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList1);
      // Undeclared exception!
      try { 
        groupImpl0.validate(writeableCommandLineImpl0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
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
  public void test12()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "rWWEKgv", 0, 0);
      linkedList0.offerLast(groupImpl0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      writeableCommandLineImpl0.addSwitch(groupImpl0, true);
      try { 
        groupImpl0.validate(writeableCommandLineImpl0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unexpected  while processing 
         //
         verifyException("org.apache.commons.cli2.option.GroupImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "{\"DKK`", 0, 833);
      linkedList0.offerLast(groupImpl0);
      LinkedList<DisplaySetting> linkedList1 = new LinkedList<DisplaySetting>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList1);
      writeableCommandLineImpl0.addSwitch(groupImpl0, false);
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
  public void test14()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, "YqA(;<G']", 0, 811);
      StringBuffer stringBuffer0 = new StringBuffer(0);
      LinkedHashSet<Object> linkedHashSet0 = new LinkedHashSet<Object>();
      Comparator<Integer> comparator0 = (Comparator<Integer>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      groupImpl0.appendUsage(stringBuffer0, (Set) linkedHashSet0, (Comparator) comparator0);
      assertEquals(811, groupImpl0.getMaximum());
      assertEquals("", stringBuffer0.toString());
      assertEquals(0, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, ".jZJ", 0, (-392));
      LinkedList<GroupImpl> linkedList1 = new LinkedList<GroupImpl>();
      linkedList1.offerLast(groupImpl0);
      linkedList1.add(groupImpl0);
      GroupImpl groupImpl1 = new GroupImpl(linkedList1, (String) null, (String) null, 811, (-506));
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
  public void test16()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", (-2673), (-2673));
      LinkedHashSet<Object> linkedHashSet0 = new LinkedHashSet<Object>();
      List list0 = groupImpl0.helpLines((-2673), linkedHashSet0, (Comparator) null);
      assertEquals((-2673), groupImpl0.getMinimum());
      assertEquals((-2673), groupImpl0.getMaximum());
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "YqA(;<G']", "YqA(;<G']", 842, 842);
      LinkedList<GroupImpl> linkedList1 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl1 = new GroupImpl(linkedList1, "YqA(;<G']", "YqA(;<G']", 842, 842);
      linkedList1.add(groupImpl0);
      groupImpl1.findOption("YqA(;<G']");
      assertEquals(842, groupImpl1.getMaximum());
      assertEquals(842, groupImpl1.getMinimum());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", (-2673), (-2673));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      groupImpl0.defaults(writeableCommandLineImpl0);
      assertEquals((-2673), groupImpl0.getMinimum());
      assertEquals((-2673), groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
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
