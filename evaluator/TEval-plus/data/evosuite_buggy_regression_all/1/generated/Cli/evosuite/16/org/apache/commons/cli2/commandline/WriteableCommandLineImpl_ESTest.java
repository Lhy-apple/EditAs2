/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 12:59:21 GMT 2023
 */

package org.apache.commons.cli2.commandline;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.commandline.WriteableCommandLineImpl;
import org.apache.commons.cli2.option.ArgumentImpl;
import org.apache.commons.cli2.option.Command;
import org.apache.commons.cli2.option.GroupImpl;
import org.apache.commons.cli2.option.PropertyOption;
import org.apache.commons.cli2.option.SourceDestArgument;
import org.apache.commons.cli2.validation.ClassValidator;
import org.apache.commons.cli2.validation.DateValidator;
import org.apache.commons.cli2.validation.FileValidator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class WriteableCommandLineImpl_ESTest extends WriteableCommandLineImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      String string0 = writeableCommandLineImpl0.getProperty("-D");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      Set set0 = writeableCommandLineImpl0.getOptionTriggers();
      assertTrue(set0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Boolean> linkedList0 = new LinkedList<Boolean>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addProperty("-D", "Passes properties and values to the application");
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      Set set0 = writeableCommandLineImpl0.getProperties();
      assertEquals(0, set0.size());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      List list0 = writeableCommandLineImpl0.getOptions();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      List list0 = writeableCommandLineImpl0.getNormalised();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      Object object0 = writeableCommandLineImpl0.getValue("-D", (Object) propertyOption0);
      assertSame(propertyOption0, object0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      DateValidator dateValidator0 = DateValidator.getTimeInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("Passes properties and values to the application", "-D", (-1139), (-1139), 'y', 'y', dateValidator0, "-D", (List) null, (-1139));
      writeableCommandLineImpl0.addValue(argumentImpl0, "-DPasses properties and values to the application");
      assertEquals((-1139), argumentImpl0.getId());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addValue(propertyOption0, propertyOption0);
      writeableCommandLineImpl0.addValue(propertyOption0, propertyOption0);
      assertFalse(propertyOption0.isRequired());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<SourceDestArgument> linkedList0 = new LinkedList<SourceDestArgument>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addSwitch(propertyOption0, false);
      // Undeclared exception!
      try { 
        writeableCommandLineImpl0.addSwitch(propertyOption0, false);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Switch already set.
         //
         verifyException("org.apache.commons.cli2.commandline.WriteableCommandLineImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = writeableCommandLineImpl0.hasOption("Passes properties and values to the application");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption((String) null, "", 1129);
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addOption(propertyOption0);
      boolean boolean0 = writeableCommandLineImpl0.hasOption((Option) propertyOption0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      DateValidator dateValidator0 = DateValidator.getTimeInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("Passes properties and values to the application", "-D", (-1139), (-1139), 'y', 'y', dateValidator0, "-D", (List) null, (-1139));
      List list0 = writeableCommandLineImpl0.getValues((Option) argumentImpl0, (List) null);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<Command> linkedList0 = new LinkedList<Command>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      FileValidator fileValidator0 = FileValidator.getExistingDirectoryInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("-D", "-D<property>=<value>", 1823, 1823, 'o', 'o', fileValidator0, "0>h@", linkedList0, (-2794));
      argumentImpl0.defaultValues(writeableCommandLineImpl0, propertyOption0);
      Object object0 = writeableCommandLineImpl0.getValue((Option) propertyOption0, (Object) null);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      linkedList0.offerFirst(propertyOption0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addValue(propertyOption0, linkedList0);
      List list0 = writeableCommandLineImpl0.getValues((Option) propertyOption0, (List) linkedList0);
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<SourceDestArgument> linkedList0 = new LinkedList<SourceDestArgument>();
      ClassValidator classValidator0 = new ClassValidator();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addValue(propertyOption0, classValidator0);
      List list0 = writeableCommandLineImpl0.getUndefaultedValues(propertyOption0);
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      List list0 = writeableCommandLineImpl0.getUndefaultedValues(propertyOption0);
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "", "-D", (-1160), 869);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      writeableCommandLineImpl0.addSwitch(groupImpl0, true);
      Boolean boolean0 = Boolean.valueOf("--");
      Boolean boolean1 = writeableCommandLineImpl0.getSwitch((Option) groupImpl0, boolean0);
      assertTrue(boolean1);
      assertNotNull(boolean1);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<SourceDestArgument> linkedList0 = new LinkedList<SourceDestArgument>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      Boolean boolean0 = Boolean.valueOf("Passes properties and values to the application");
      Boolean boolean1 = writeableCommandLineImpl0.getSwitch((Option) propertyOption0, boolean0);
      assertFalse(boolean1);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<SourceDestArgument> linkedList0 = new LinkedList<SourceDestArgument>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      Boolean boolean0 = writeableCommandLineImpl0.getSwitch((Option) propertyOption0);
      assertNull(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      FileValidator fileValidator0 = FileValidator.getExistingDirectoryInstance();
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("\"", "\"", 0, 0, 'p', 's', fileValidator0, "--", linkedList0, 1);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(sourceDestArgument0, linkedList0);
      writeableCommandLineImpl0.addProperty((Option) sourceDestArgument0, "--", "\"");
      writeableCommandLineImpl0.addProperty((Option) sourceDestArgument0, "--", "--");
      assertEquals(0, sourceDestArgument0.getId());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addProperty((Option) propertyOption0, "-D", "Passes properties and values to the application");
      String string0 = writeableCommandLineImpl0.getProperty((Option) propertyOption0, "xhAu");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addProperty((Option) propertyOption0, "-D", "Passes properties and values to the application");
      Set set0 = writeableCommandLineImpl0.getProperties((Option) propertyOption0);
      assertEquals(1, set0.size());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = writeableCommandLineImpl0.looksLikeOption("Passes properties and values to the application");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = writeableCommandLineImpl0.looksLikeOption("-D");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addValue(propertyOption0, "-D");
      List list0 = writeableCommandLineImpl0.getValues((Option) propertyOption0, (List) linkedList0);
      assertNotNull(list0);
      
      WriteableCommandLineImpl writeableCommandLineImpl1 = new WriteableCommandLineImpl(propertyOption0, list0);
      String string0 = writeableCommandLineImpl1.toString();
      assertEquals("-D", string0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addValue(propertyOption0, "Iz#}[:XS{bGHz V");
      List list0 = writeableCommandLineImpl0.getValues((Option) propertyOption0, (List) linkedList0);
      WriteableCommandLineImpl writeableCommandLineImpl1 = new WriteableCommandLineImpl(propertyOption0, list0);
      String string0 = writeableCommandLineImpl1.toString();
      assertEquals("\"Iz#}[:XS{bGHz V\"", string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      linkedList0.offerFirst(propertyOption0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      linkedList0.offerFirst(propertyOption0);
      writeableCommandLineImpl0.addValue(propertyOption0, "Iz#}[:XS{bGHz V");
      List list0 = writeableCommandLineImpl0.getValues((Option) propertyOption0, (List) linkedList0);
      WriteableCommandLineImpl writeableCommandLineImpl1 = new WriteableCommandLineImpl(propertyOption0, list0);
      // Undeclared exception!
      try { 
        writeableCommandLineImpl1.toString();
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // org.apache.commons.cli2.option.PropertyOption cannot be cast to java.lang.String
         //
         verifyException("org.apache.commons.cli2.commandline.WriteableCommandLineImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.setDefaultValues(propertyOption0, (List) null);
      assertFalse(propertyOption0.isRequired());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      Boolean boolean0 = Boolean.TRUE;
      writeableCommandLineImpl0.setDefaultSwitch(propertyOption0, boolean0);
      assertEquals(68, propertyOption0.getId());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.setDefaultSwitch(propertyOption0, (Boolean) null);
      assertFalse(propertyOption0.isRequired());
  }
}
