/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 01:56:51 GMT 2023
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
import org.apache.commons.cli2.option.DefaultOption;
import org.apache.commons.cli2.option.PropertyOption;
import org.apache.commons.cli2.validation.DateValidator;
import org.apache.commons.cli2.validation.FileValidator;
import org.apache.commons.cli2.validation.Validator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class WriteableCommandLineImpl_ESTest extends WriteableCommandLineImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Boolean> linkedList0 = new LinkedList<Boolean>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      String string0 = writeableCommandLineImpl0.getProperty("Passes properties and values to the application", "-D");
      assertEquals("-D", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      FileValidator fileValidator0 = FileValidator.getExistingDirectoryInstance();
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("Di9", "-DPasses properties and values to the application", 2126, 2147483645, ';', ';', fileValidator0, "-DPasses properties and values to the application", linkedList0, (-191006455));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(argumentImpl0, linkedList0);
      Set set0 = writeableCommandLineImpl0.getOptionTriggers();
      assertTrue(set0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      LinkedList<Boolean> linkedList0 = new LinkedList<Boolean>();
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addProperty("Passes properties and values to the application", "Passes properties and values to the application");
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      FileValidator fileValidator0 = FileValidator.getExistingDirectoryInstance();
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("Di9", "-DPasses properties and values to the application", 2126, 2147483645, ';', ';', fileValidator0, "-DPasses properties and values to the application", linkedList0, (-191006455));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(argumentImpl0, linkedList0);
      Set set0 = writeableCommandLineImpl0.getProperties();
      assertEquals(0, set0.size());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      LinkedList<Boolean> linkedList0 = new LinkedList<Boolean>();
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      int int0 = writeableCommandLineImpl0.getOptionCount((Option) propertyOption0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<DefaultOption> linkedList0 = new LinkedList<DefaultOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      List list0 = writeableCommandLineImpl0.getNormalised();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Boolean> linkedList0 = new LinkedList<Boolean>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = writeableCommandLineImpl0.hasOption("-D");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Boolean> linkedList0 = new LinkedList<Boolean>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addSwitch(propertyOption0, false);
      boolean boolean0 = writeableCommandLineImpl0.hasOption("-D");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      FileValidator fileValidator0 = FileValidator.getExistingDirectoryInstance();
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("Di9", "-DPasses properties and values to the application", 2126, 2147483645, ';', ';', fileValidator0, "-DPasses properties and values to the application", linkedList0, (-191006455));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(argumentImpl0, linkedList0);
      PropertyOption propertyOption0 = new PropertyOption();
      writeableCommandLineImpl0.addValue(propertyOption0, "-DPasses properties and values to the application");
      List list0 = writeableCommandLineImpl0.getValues((Option) propertyOption0, (List) linkedList0);
      WriteableCommandLineImpl writeableCommandLineImpl1 = new WriteableCommandLineImpl(propertyOption0, list0);
      String string0 = writeableCommandLineImpl1.toString();
      assertEquals("\"-DPasses properties and values to the application\"", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      LinkedList<Boolean> linkedList0 = new LinkedList<Boolean>();
      DateValidator dateValidator0 = new DateValidator();
      ArgumentImpl argumentImpl0 = new ArgumentImpl((String) null, (String) null, (-3412), (-3412), 'b', 'b', dateValidator0, (String) null, linkedList0, (-3412));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(argumentImpl0, linkedList0);
      writeableCommandLineImpl0.addValue(argumentImpl0, linkedList0);
      assertEquals("arg", argumentImpl0.getPreferredName());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      LinkedList<Boolean> linkedList0 = new LinkedList<Boolean>();
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addSwitch(propertyOption0, false);
      // Undeclared exception!
      try { 
        writeableCommandLineImpl0.addSwitch(propertyOption0, true);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Switch already set.
         //
         verifyException("org.apache.commons.cli2.commandline.WriteableCommandLineImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<DefaultOption> linkedList0 = new LinkedList<DefaultOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addSwitch(propertyOption0, true);
      assertEquals("-D", propertyOption0.getPreferredName());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      FileValidator fileValidator0 = FileValidator.getExistingDirectoryInstance();
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("Di9", "-DPasses properties and values to the application", 2126, 2147483645, ';', ';', fileValidator0, "-DPasses properties and values to the application", linkedList0, (-191006455));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(argumentImpl0, linkedList0);
      PropertyOption propertyOption0 = new PropertyOption();
      List list0 = writeableCommandLineImpl0.getValues((Option) propertyOption0, (List) linkedList0);
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      List list0 = writeableCommandLineImpl0.getValues((Option) propertyOption0, (List) null);
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Boolean> linkedList0 = new LinkedList<Boolean>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addSwitch(propertyOption0, false);
      Boolean boolean0 = writeableCommandLineImpl0.getSwitch((Option) propertyOption0, (Boolean) null);
      assertNotNull(boolean0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      LinkedList<Boolean> linkedList0 = new LinkedList<Boolean>();
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      Boolean boolean0 = writeableCommandLineImpl0.getSwitch("-D");
      assertNull(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      LinkedList<Boolean> linkedList0 = new LinkedList<Boolean>();
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = writeableCommandLineImpl0.looksLikeOption("Passes properties and values to the application");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      LinkedList<Boolean> linkedList0 = new LinkedList<Boolean>();
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = writeableCommandLineImpl0.looksLikeOption("-D");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      FileValidator fileValidator0 = FileValidator.getExistingInstance();
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("?@${:fy", "?@${:fy", ';', ';', ';', ';', fileValidator0, "\"", linkedList0, (-191006455));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(argumentImpl0, linkedList0);
      PropertyOption propertyOption0 = new PropertyOption();
      writeableCommandLineImpl0.addValue(propertyOption0, "\"");
      List list0 = writeableCommandLineImpl0.getValues((Option) propertyOption0, (List) linkedList0);
      WriteableCommandLineImpl writeableCommandLineImpl1 = new WriteableCommandLineImpl(argumentImpl0, list0);
      String string0 = writeableCommandLineImpl1.toString();
      assertEquals("\"", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      FileValidator fileValidator0 = FileValidator.getExistingInstance();
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("?@${:fy", "?@${:fy", ';', ';', ';', ';', fileValidator0, "\"", linkedList0, (-191006455));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(argumentImpl0, linkedList0);
      PropertyOption propertyOption0 = new PropertyOption();
      writeableCommandLineImpl0.addValue(propertyOption0, "\"");
      List list0 = writeableCommandLineImpl0.getValues((Option) propertyOption0, (List) linkedList0);
      writeableCommandLineImpl0.addValue(propertyOption0, linkedList0);
      WriteableCommandLineImpl writeableCommandLineImpl1 = new WriteableCommandLineImpl(argumentImpl0, list0);
      // Undeclared exception!
      try { 
        writeableCommandLineImpl1.toString();
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // java.util.LinkedList cannot be cast to java.lang.String
         //
         verifyException("org.apache.commons.cli2.commandline.WriteableCommandLineImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<Boolean> linkedList0 = new LinkedList<Boolean>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.setDefaultValues(propertyOption0, linkedList0);
      assertEquals(0, linkedList0.size());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Boolean> linkedList0 = new LinkedList<Boolean>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      ArgumentImpl argumentImpl0 = new ArgumentImpl("%3?=hh[U*b", "-D", 37, 37, 'S', 'm', (Validator) null, "Passes properties and values to the application", (List) null, 37);
      writeableCommandLineImpl0.setDefaultValues(argumentImpl0, (List) null);
      assertEquals(37, argumentImpl0.getId());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<Boolean> linkedList0 = new LinkedList<Boolean>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      Boolean boolean0 = Boolean.valueOf("-D");
      writeableCommandLineImpl0.setDefaultSwitch(propertyOption0, boolean0);
      assertEquals("Passes properties and values to the application", propertyOption0.getDescription());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Boolean> linkedList0 = new LinkedList<Boolean>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.setDefaultSwitch(propertyOption0, (Boolean) null);
      assertEquals("Passes properties and values to the application", propertyOption0.getDescription());
  }
}