/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:04:03 GMT 2023
 */

package org.apache.commons.cli2.option;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.StringWriter;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Set;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.WriteableCommandLine;
import org.apache.commons.cli2.commandline.WriteableCommandLineImpl;
import org.apache.commons.cli2.option.ArgumentImpl;
import org.apache.commons.cli2.option.Command;
import org.apache.commons.cli2.option.DefaultOption;
import org.apache.commons.cli2.option.GroupImpl;
import org.apache.commons.cli2.option.PropertyOption;
import org.apache.commons.cli2.option.SourceDestArgument;
import org.apache.commons.cli2.option.Switch;
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
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "-D", "-D", 93, 93);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.getSwitch((Option) groupImpl0, (Boolean) null);
      assertEquals(93, groupImpl0.getMinimum());
      assertEquals(93, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      LinkedList<String> linkedList0 = new LinkedList<String>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Unexpected.token", "org.apache.commons.cli2.validation.UrlValidator", (-2239), 1380);
      groupImpl0.getAnonymous();
      assertEquals("Unexpected.token", groupImpl0.getPreferredName());
      assertEquals(1380, groupImpl0.getMaximum());
      assertEquals("org.apache.commons.cli2.validation.UrlValidator", groupImpl0.getDescription());
      assertEquals((-2239), groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      LinkedList<Command> linkedList0 = new LinkedList<Command>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "L/-w+Y&r", "L/-w+Y&r", (-507), (-507));
      int int0 = groupImpl0.getMaximum();
      assertEquals((-507), int0);
      assertEquals((-507), groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "-D", "-D", 83, 83);
      groupImpl0.findOption("|");
      assertEquals(1, linkedList0.size());
      assertTrue(groupImpl0.isRequired());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      DateValidator dateValidator0 = DateValidator.getDateTimeInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("=IE", "=IE", 0, 3399, '`', '`', dateValidator0, "", linkedList0, (-3));
      linkedList0.add(argumentImpl0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "=IE", "=IE", 0, 0);
      groupImpl0.toString();
      assertFalse(linkedList0.contains(argumentImpl0));
      assertEquals(0, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "-D", "Passes properties and values to the application", (-1001), (-1001));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "-D (-D<property>=<value>)");
      assertEquals("Passes properties and values to the application", groupImpl0.getDescription());
      assertEquals("-D", groupImpl0.getPreferredName());
      assertEquals((-1001), groupImpl0.getMaximum());
      assertEquals((-1001), groupImpl0.getMinimum());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "aHS+Y7K}CN-Y", "aHS+Y7K}CN-Y", 3591, 3591);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, (String) null);
      assertEquals(3591, groupImpl0.getMinimum());
      assertEquals(3591, groupImpl0.getMaximum());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "-D", "-D", 93, 93);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "-D");
      assertEquals(1, linkedList0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "-D", "Passes properties and values to the application", (-1001), (-1001));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "-D (-D<property>=<value>)");
      assertEquals(1, linkedList0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "-D", "-D", 93, 93);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "Passes properties and values to the application");
      assertEquals(1, linkedList0.size());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      FileValidator fileValidator0 = new FileValidator();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("o_ZFnp7", "ej\"[JwOb_!?", 2147483645, 2147483645, 'x', '_', fileValidator0, "nfPoS", linkedList0, 1487);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0, '5', '\u0000', "nfPoS", linkedList0);
      linkedList0.offerLast(sourceDestArgument0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "org.apache.commons.cli2.validation.DateValidator", "org.apache.commons.cli2.validation.DateValidator", 2, 2);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, "org.apache.commons.cli2.validation.DateValidator");
      assertEquals(0, linkedList0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "*R<R6(8J|.``Q", "*R<R6(8J|.``Q", (-24), (-24));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      ListIterator<Object> listIterator0 = linkedList0.listIterator();
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertEquals((-24), groupImpl0.getMinimum());
      assertEquals((-24), groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "]", "2C=*:", 0, 0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      LinkedList<String> linkedList1 = new LinkedList<String>();
      linkedList1.add("");
      ListIterator<String> listIterator0 = linkedList1.listIterator(0);
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertEquals(0, groupImpl0.getMaximum());
      assertEquals("]", groupImpl0.getPreferredName());
      assertEquals(0, groupImpl0.getMinimum());
      assertEquals("2C=*:", groupImpl0.getDescription());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "-D", (-1), (-265));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, (List) null);
      ListIterator<GroupImpl> listIterator0 = (ListIterator<GroupImpl>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true).when(listIterator0).hasNext();
      doReturn((Object) null).when(listIterator0).next();
      doReturn((Object) null).when(listIterator0).previous();
      groupImpl0.process(writeableCommandLineImpl0, listIterator0);
      assertEquals((-265), groupImpl0.getMaximum());
      assertEquals("-D", groupImpl0.getDescription());
      assertEquals("Passes properties and values to the application", groupImpl0.getPreferredName());
      assertEquals((-1), groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "-D", (-1), (-265));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, (List) null);
      groupImpl0.validate(writeableCommandLineImpl0);
      assertEquals(1, linkedList0.size());
      assertEquals((-1), groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "_vKpCH D<z/!'k741", "_vKpCH D<z/!'k741", 947, 61);
      linkedList0.addLast(groupImpl0);
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
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "-D", "-D", 93, 93);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addSwitch(propertyOption0, true);
      try { 
        groupImpl0.validate(writeableCommandLineImpl0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing option -D
         //
         verifyException("org.apache.commons.cli2.option.GroupImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "-D", (-1), (-265));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addSwitch(propertyOption0, true);
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
  public void test18()  throws Throwable  {
      LinkedList<Switch> linkedList0 = new LinkedList<Switch>();
      NumberValidator numberValidator0 = NumberValidator.getNumberInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("g}~", "", 41, 41, '\\', '\\', numberValidator0, "|", linkedList0, 1979);
      LinkedList<ArgumentImpl> linkedList1 = new LinkedList<ArgumentImpl>();
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0, '\u0000', 'v', (String) null, linkedList1);
      linkedList1.add((ArgumentImpl) sourceDestArgument0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList1, "Unexpected.token", "Unexpected.token", (-1754), '\\');
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(argumentImpl0, linkedList1);
      try { 
        groupImpl0.validate(writeableCommandLineImpl0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing value(s) g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~ g}~
         //
         verifyException("org.apache.commons.cli2.option.ArgumentImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", 0, 0);
      LinkedHashSet<GroupImpl> linkedHashSet0 = new LinkedHashSet<GroupImpl>();
      StringBuffer stringBuffer0 = new StringBuffer();
      groupImpl0.appendUsage(stringBuffer0, (Set) linkedHashSet0, (Comparator) null, "");
      assertEquals(0, groupImpl0.getMinimum());
      assertEquals("", stringBuffer0.toString());
      assertEquals(0, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "=IE", "=IE", 0, 0);
      String string0 = groupImpl0.toString();
      assertEquals(0, groupImpl0.getMaximum());
      assertEquals("[=IE ()]", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      LinkedList<Switch> linkedList0 = new LinkedList<Switch>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, 93, 93);
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
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, ";pdDC~N)#A`FH(Gk2a", 34, 91);
      StringWriter stringWriter0 = new StringWriter();
      StringBuffer stringBuffer0 = stringWriter0.getBuffer();
      LinkedHashSet<PropertyOption> linkedHashSet0 = new LinkedHashSet<PropertyOption>();
      Comparator<DefaultOption> comparator0 = (Comparator<DefaultOption>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      groupImpl0.appendUsage(stringBuffer0, (Set) linkedHashSet0, (Comparator) comparator0, (String) null);
      assertEquals("", stringBuffer0.toString());
      assertEquals(34, groupImpl0.getMinimum());
      assertEquals(91, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add(propertyOption0);
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "-D", "-D", 88, 88);
      String string0 = groupImpl0.toString();
      assertEquals(2, linkedList0.size());
      assertEquals("-D (-D<property>=<value>|-D<property>=<value>)", string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "", 0, 0);
      LinkedHashSet<GroupImpl> linkedHashSet0 = new LinkedHashSet<GroupImpl>();
      List list0 = groupImpl0.helpLines(0, linkedHashSet0, (Comparator) null);
      assertEquals(0, groupImpl0.getMinimum());
      assertEquals(0, list0.size());
      assertEquals(0, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "-D", "-D", 93, 93);
      groupImpl0.findOption("-D");
      assertEquals(1, linkedList0.size());
      assertEquals(93, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "*R<R6(8J|.``Q", "*R<R6(8J|.``Q", 0, 0);
      boolean boolean0 = groupImpl0.isRequired();
      assertEquals(0, groupImpl0.getMinimum());
      assertEquals(0, groupImpl0.getMaximum());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      PropertyOption propertyOption0 = new PropertyOption();
      linkedList0.add(propertyOption0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "-D", "-D", 93, 93);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      groupImpl0.defaults(writeableCommandLineImpl0);
      assertEquals(1, linkedList0.size());
      assertEquals(93, groupImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      FileValidator fileValidator0 = FileValidator.getExistingFileInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("", "", 32, 32, '`', '`', fileValidator0, "", linkedList0, 91);
      linkedList0.add((Object) argumentImpl0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "W=IU}u$e=Q>od?L", "}XhFU|>Q,", 91, 32);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      groupImpl0.defaults(writeableCommandLineImpl0);
      assertEquals(0, linkedList0.size());
  }
}