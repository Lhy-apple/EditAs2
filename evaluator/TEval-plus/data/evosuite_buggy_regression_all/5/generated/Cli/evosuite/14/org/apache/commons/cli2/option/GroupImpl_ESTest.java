/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:27:05 GMT 2023
 */

package org.apache.commons.cli2.option;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.nio.charset.Charset;
import java.text.NumberFormat;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Locale;
import java.util.Set;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.WriteableCommandLine;
import org.apache.commons.cli2.commandline.WriteableCommandLineImpl;
import org.apache.commons.cli2.option.ArgumentImpl;
import org.apache.commons.cli2.option.Command;
import org.apache.commons.cli2.option.GroupImpl;
import org.apache.commons.cli2.option.PropertyOption;
import org.apache.commons.cli2.validation.DateValidator;
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
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "x", "x", Integer.MAX_VALUE, Integer.MAX_VALUE);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      writeableCommandLineImpl0.getValues((Option) groupImpl0);
      assertEquals(Integer.MAX_VALUE, groupImpl0.getMaximum());
      assertEquals(Integer.MAX_VALUE, groupImpl0.getMinimum());
      assertTrue(groupImpl0.isRequired());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "$!\u0005?K+=,?l\"Au+w", "$!\u0005?K+=,?l\"Au+w", 0, 0);
      groupImpl0.getAnonymous();
      assertEquals(0, groupImpl0.getMinimum());
      assertEquals(0, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "$!\u0005?Kb=,?l\"Au5>w", "$!\u0005?Kb=,?l\"Au5>w", 0, 0);
      int int0 = groupImpl0.getMaximum();
      assertEquals(0, groupImpl0.getMinimum());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      FileValidator fileValidator0 = FileValidator.getExistingFileInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("Unexpected.token", "Unexpected.token", 0, 0, ' ', ' ', fileValidator0, "Unexpected.token", linkedList0, 0);
      linkedList0.offerLast(argumentImpl0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Unexpected.token", "Unexpected.token", 0, 0);
      String string0 = groupImpl0.toString();
      assertFalse(linkedList0.contains(argumentImpl0));
      assertEquals("[Unexpected.token ()] ", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "{Y[<Y%X) <^y", (String) null, 0, 0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, (String) null);
      assertEquals(0, groupImpl0.getMinimum());
      assertFalse(boolean0);
      assertEquals(0, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add((Object) propertyOption0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "", 41, 41);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "-D");
      assertEquals(1, linkedList0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", 0, 0);
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add((Object) propertyOption0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      GroupImpl groupImpl1 = new GroupImpl(linkedList0, "", (String) null, 0, (-2612));
      boolean boolean0 = groupImpl1.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "Passes properties and values to the application");
      assertEquals(1, linkedList0.size());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", 0, 0);
      PropertyOption propertyOption0 = new PropertyOption("", "", 0);
      linkedList0.add((Object) propertyOption0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      GroupImpl groupImpl1 = new GroupImpl(linkedList0, "bM`//qp#wh(B1Uz.", "Passes properties and values to the application", 0, (-2612));
      boolean boolean0 = groupImpl1.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "Passes properties and values to the application");
      assertEquals(1, linkedList0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", 0, 0);
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "-D");
      assertEquals(0, groupImpl0.getMaximum());
      assertFalse(boolean0);
      assertEquals(0, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      DateValidator dateValidator0 = new DateValidator(linkedList0);
      ArgumentImpl argumentImpl0 = new ArgumentImpl("-D", "Passes properties and values to the application", (-1065), (-1065), 'Q', 'Q', dateValidator0, "-D", linkedList0, 'Q');
      linkedList0.add((Object) argumentImpl0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", (-1065), (-1065));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "--");
      assertEquals(0, linkedList0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "huS1wz{", "huS1wz{", 0, 0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      ListIterator<Object> listIterator0 = linkedList0.listIterator(0);
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertEquals(0, groupImpl0.getMinimum());
      assertEquals(0, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "$!\u0005?Kb=,?l\"Au5>w", "$!\u0005?Kb=,?l\"Au5>w", 0, 0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      ListIterator<Integer> listIterator0 = (ListIterator<Integer>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true).when(listIterator0).hasNext();
      doReturn((Object) null).when(listIterator0).next();
      doReturn((Object) null).when(listIterator0).previous();
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertEquals(0, groupImpl0.getMinimum());
      assertEquals(0, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      LinkedList<Command> linkedList0 = new LinkedList<Command>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Unexpectedken", "Unexpectedken", 0, 0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      Charset charset0 = Charset.defaultCharset();
      Set<String> set0 = charset0.aliases();
      LinkedList<String> linkedList1 = new LinkedList<String>(set0);
      ListIterator<String> listIterator0 = linkedList1.listIterator();
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertEquals(0, groupImpl0.getMinimum());
      assertEquals(0, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      LinkedList<Command> linkedList0 = new LinkedList<Command>();
      LinkedHashSet<Command> linkedHashSet0 = new LinkedHashSet<Command>();
      Locale locale0 = new Locale("DateValidator.date.OutOfRange", "DateValidator.date.OutOfRange");
      NumberFormat numberFormat0 = NumberFormat.getInstance(locale0);
      NumberValidator numberValidator0 = new NumberValidator(numberFormat0);
      ArgumentImpl argumentImpl0 = new ArgumentImpl("(p![):", "(p![):", (-2147), (-2147), '[', '[', numberValidator0, "Missing.option", linkedList0, (-2147));
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "(p![):", "Missing.option", (-2147), (-1));
      Command command0 = new Command("(p![):", "(p![):", linkedHashSet0, true, argumentImpl0, groupImpl0, (-1));
      linkedList0.add(command0);
      GroupImpl groupImpl1 = new GroupImpl(linkedList0, "(p![):", "(p![):", 1388, 1388);
      ListIterator<String> listIterator0 = (ListIterator<String>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true).when(listIterator0).hasNext();
      doReturn("(p![):", (Object) null).when(listIterator0).next();
      doReturn("(p![):").when(listIterator0).previous();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl1, linkedList0);
      // Undeclared exception!
      try { 
        groupImpl1.process(writeableCommandLineImpl0, listIterator0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.cli2.option.ParentImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      LinkedList<Command> linkedList0 = new LinkedList<Command>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "()vq0", "()vq0", 1388, 1388);
      ListIterator<String> listIterator0 = (ListIterator<String>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true).when(listIterator0).hasNext();
      doReturn("()vq0").when(listIterator0).next();
      doReturn("()vq0").when(listIterator0).previous();
      PropertyOption propertyOption0 = new PropertyOption("()vq0", "0BD", 1388);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertEquals(1388, groupImpl0.getMaximum());
      assertEquals(1388, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add((Object) propertyOption0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "-D", 0, 0);
      groupImpl0.validate(writeableCommandLineImpl0);
      assertEquals(1, linkedList0.size());
      assertEquals(0, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Pgcx%~>IU^nIV", "Pgcx%~>IU^nIV", 28, 28);
      linkedList0.add((Object) groupImpl0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
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
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "$!\u0005?Kb=,?l\"Au5>w", "$!\u0005?Kb=,?l\"Au5>w", 0, 0);
      linkedList0.add((Object) groupImpl0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
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
  public void test18()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add((Object) propertyOption0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addSwitch(propertyOption0, true);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "-D", 0, 0);
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
  public void test19()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add((Object) propertyOption0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addSwitch(propertyOption0, true);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "H3JwQ_!}", "Passes properties and values to the application", 927, 93);
      try { 
        groupImpl0.validate(writeableCommandLineImpl0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing option H3JwQ_!}
         //
         verifyException("org.apache.commons.cli2.option.GroupImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      PropertyOption propertyOption0 = new PropertyOption();
      DateValidator dateValidator0 = new DateValidator(linkedList0);
      ArgumentImpl argumentImpl0 = new ArgumentImpl("5C},s``-3gZ9", "Passes properties and values to the application", (-1065), (-1065), 'Q', '7', dateValidator0, "5C},s``-3gZ9", linkedList0, (-1065));
      linkedList0.add((Object) argumentImpl0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", (-1065), (-1065));
      // Undeclared exception!
      try { 
        groupImpl0.validate(writeableCommandLineImpl0);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: -1065
         //
         verifyException("java.util.Collections$EmptyList", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, 5, 1464);
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
  public void test22()  throws Throwable  {
      LinkedList<Command> linkedList0 = new LinkedList<Command>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, "!(9%", 0, 0);
      LinkedHashSet<Integer> linkedHashSet0 = new LinkedHashSet<Integer>();
      StringBuffer stringBuffer0 = new StringBuffer();
      Comparator<Object> comparator0 = (Comparator<Object>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      groupImpl0.appendUsage(stringBuffer0, (Set) linkedHashSet0, (Comparator) comparator0, "[Qw18t[8$c_");
      assertEquals(0, groupImpl0.getMaximum());
      assertEquals(0, groupImpl0.getMinimum());
      assertEquals(0, stringBuffer0.length());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add((Object) propertyOption0);
      linkedList0.offerFirst(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Unexpected.token", "t3<+h]9K=j&8", 0, 0);
      String string0 = groupImpl0.toString();
      assertEquals(2, linkedList0.size());
      assertEquals("[Unexpected.token (-D<property>=<value>|-D<property>=<value>)]", string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", (-1065), (-1065));
      LinkedHashSet<Integer> linkedHashSet0 = new LinkedHashSet<Integer>();
      Comparator<Integer> comparator0 = (Comparator<Integer>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      List list0 = groupImpl0.helpLines('\u0000', linkedHashSet0, comparator0);
      assertEquals((-1065), groupImpl0.getMaximum());
      assertEquals((-1065), groupImpl0.getMinimum());
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", 10, 10);
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add((Object) propertyOption0);
      groupImpl0.findOption("Passes properties and values to the application");
      assertEquals(10, groupImpl0.getMinimum());
      assertEquals(10, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", 10, 10);
      PropertyOption propertyOption0 = new PropertyOption("Passes properties and values to the application", "-D", 10);
      linkedList0.add((Object) propertyOption0);
      Option option0 = groupImpl0.findOption("Passes properties and values to the application");
      assertEquals(10, groupImpl0.getMinimum());
      assertEquals(10, groupImpl0.getMaximum());
      assertNotNull(option0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", 0, 0);
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add((Object) propertyOption0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      groupImpl0.defaults(writeableCommandLineImpl0);
      assertEquals(0, groupImpl0.getMinimum());
      assertEquals(0, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      FileValidator fileValidator0 = FileValidator.getExistingFileInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("Unexpected.token", "Unexpected.token", 0, 0, ' ', ' ', fileValidator0, "Unexpected.token", linkedList0, 0);
      linkedList0.offerLast(argumentImpl0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Unexpected.token", "Unexpected.token", 0, 0);
      // Undeclared exception!
      try { 
        groupImpl0.defaults((WriteableCommandLine) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.cli2.option.ArgumentImpl", e);
      }
  }
}
