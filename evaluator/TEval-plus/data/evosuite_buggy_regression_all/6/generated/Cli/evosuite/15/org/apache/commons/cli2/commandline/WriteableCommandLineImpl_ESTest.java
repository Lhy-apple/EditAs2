/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:48:30 GMT 2023
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
import org.apache.commons.cli2.option.GroupImpl;
import org.apache.commons.cli2.option.PropertyOption;
import org.apache.commons.cli2.option.Switch;
import org.apache.commons.cli2.validation.FileValidator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class WriteableCommandLineImpl_ESTest extends WriteableCommandLineImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      String string0 = writeableCommandLineImpl0.getProperty("-D");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<Switch> linkedList0 = new LinkedList<Switch>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      Set set0 = writeableCommandLineImpl0.getOptionTriggers();
      assertEquals(0, set0.size());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      Set set0 = writeableCommandLineImpl0.getProperties();
      assertEquals(0, set0.size());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      List list0 = writeableCommandLineImpl0.getOptions();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      List list0 = writeableCommandLineImpl0.getNormalised();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<DefaultOption> linkedList0 = new LinkedList<DefaultOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = writeableCommandLineImpl0.hasOption("Passes properties and values to the application");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      FileValidator fileValidator0 = FileValidator.getExistingFileInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("Option.trigger.needs.prefix", "Passes properties and values to the application", 1797, 1797, 'U', 'U', fileValidator0, "Passes properties and values to the application", linkedList0, (-96447404));
      writeableCommandLineImpl0.addValue(argumentImpl0, fileValidator0);
      assertEquals("Option.trigger.needs.prefix", argumentImpl0.getPreferredName());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<DefaultOption> linkedList0 = new LinkedList<DefaultOption>();
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
  public void test08()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addSwitch(propertyOption0, true);
      assertEquals(68, propertyOption0.getId());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addOption(propertyOption0);
      boolean boolean0 = writeableCommandLineImpl0.hasOption((Option) propertyOption0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<DefaultOption> linkedList0 = new LinkedList<DefaultOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      List list0 = writeableCommandLineImpl0.getValues((Option) propertyOption0, (List) linkedList0);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      List list0 = writeableCommandLineImpl0.getValues((Option) propertyOption0, (List) null);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      Boolean boolean0 = Boolean.TRUE;
      writeableCommandLineImpl0.addValue(propertyOption0, boolean0);
      List list0 = writeableCommandLineImpl0.getUndefaultedValues(propertyOption0);
      assertFalse(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      List list0 = writeableCommandLineImpl0.getUndefaultedValues(propertyOption0);
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addSwitch(propertyOption0, false);
      Boolean boolean0 = writeableCommandLineImpl0.getSwitch((Option) propertyOption0);
      assertNotNull(boolean0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<DefaultOption> linkedList0 = new LinkedList<DefaultOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      Boolean boolean0 = writeableCommandLineImpl0.getSwitch("-D");
      assertNull(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<DefaultOption> linkedList0 = new LinkedList<DefaultOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addProperty((Option) propertyOption0, "Passes properties and values to the application", "-D");
      writeableCommandLineImpl0.addProperty("Passes properties and values to the application", "-D");
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addProperty((Option) propertyOption0, "Cl", "-DPasses properties and values to the application");
      String string0 = writeableCommandLineImpl0.getProperty((Option) propertyOption0, "Cl");
      assertEquals("-DPasses properties and values to the application", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addProperty((Option) propertyOption0, "-D", "-Dj%J\"HoW_V<+;U,gd");
      Set set0 = writeableCommandLineImpl0.getProperties();
      assertFalse(set0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = writeableCommandLineImpl0.looksLikeOption("Passes properties and values to the application");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = writeableCommandLineImpl0.looksLikeOption("-D");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      LinkedList<Integer> linkedList1 = new LinkedList<Integer>();
      writeableCommandLineImpl0.addValue(propertyOption0, "Passes properties and values to the application");
      List list0 = writeableCommandLineImpl0.getValues((Option) propertyOption0, (List) linkedList1);
      WriteableCommandLineImpl writeableCommandLineImpl1 = new WriteableCommandLineImpl(propertyOption0, list0);
      String string0 = writeableCommandLineImpl1.toString();
      assertEquals("\"Passes properties and values to the application\"", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      LinkedList<Integer> linkedList1 = new LinkedList<Integer>();
      writeableCommandLineImpl0.addValue(propertyOption0, "Cl");
      List list0 = writeableCommandLineImpl0.getValues((Option) propertyOption0, (List) linkedList1);
      WriteableCommandLineImpl writeableCommandLineImpl1 = new WriteableCommandLineImpl(propertyOption0, list0);
      Integer integer0 = new Integer(32);
      writeableCommandLineImpl0.addValue(propertyOption0, integer0);
      // Undeclared exception!
      try { 
        writeableCommandLineImpl1.toString();
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // java.lang.Integer cannot be cast to java.lang.String
         //
         verifyException("org.apache.commons.cli2.commandline.WriteableCommandLineImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.setDefaultValues(propertyOption0, linkedList0);
      assertFalse(propertyOption0.isRequired());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      writeableCommandLineImpl0.setDefaultValues(propertyOption0, (List) null);
      assertEquals("Passes properties and values to the application", propertyOption0.getDescription());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      Boolean boolean0 = Boolean.TRUE;
      writeableCommandLineImpl0.setDefaultSwitch(propertyOption0, boolean0);
      assertEquals(68, propertyOption0.getId());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.setDefaultSwitch(propertyOption0, (Boolean) null);
      assertEquals("Passes properties and values to the application", propertyOption0.getDescription());
  }
}