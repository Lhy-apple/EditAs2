/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 17:45:10 GMT 2023
 */

package org.apache.commons.cli2.commandline;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedList;
import java.util.List;
import java.util.Properties;
import java.util.Set;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.commandline.WriteableCommandLineImpl;
import org.apache.commons.cli2.option.ArgumentImpl;
import org.apache.commons.cli2.option.DefaultOption;
import org.apache.commons.cli2.option.PropertyOption;
import org.apache.commons.cli2.validation.NumberValidator;
import org.apache.commons.cli2.validation.Validator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class WriteableCommandLineImpl_ESTest extends WriteableCommandLineImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      String string0 = writeableCommandLineImpl0.getProperty("Passes properties and values to the application");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      Set set0 = writeableCommandLineImpl0.getOptionTriggers();
      assertTrue(set0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      writeableCommandLineImpl0.addProperty("Passes properties and values to the application", "Passes properties and values to the application");
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      Set set0 = writeableCommandLineImpl0.getProperties();
      assertEquals(0, set0.size());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Boolean> linkedList0 = new LinkedList<Boolean>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      List list0 = writeableCommandLineImpl0.getOptions();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      List list0 = writeableCommandLineImpl0.getNormalised();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      Object object0 = writeableCommandLineImpl0.getValue("Passes properties and values to the application");
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<DefaultOption> linkedList0 = new LinkedList<DefaultOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addOption(propertyOption0);
      boolean boolean0 = writeableCommandLineImpl0.hasOption("-D");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      writeableCommandLineImpl0.addValue(propertyOption0, propertyOption0);
      writeableCommandLineImpl0.addValue(propertyOption0, (Object) null);
      assertEquals(68, propertyOption0.getId());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      NumberValidator numberValidator0 = NumberValidator.getNumberInstance();
      LinkedList<ArgumentImpl> linkedList0 = new LinkedList<ArgumentImpl>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("Option.trigger.needs.prefix", "Option.trigger.needs.prefix", 32, 32, 'n', 'n', numberValidator0, "Option.trigger.needs.prefix", linkedList0, 32);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(argumentImpl0, linkedList0);
      writeableCommandLineImpl0.addValue(argumentImpl0, "Option.trigger.needs.prefix");
      assertEquals(32, argumentImpl0.getMaximum());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("Switch.already.set", " ;Xmbna4$L#", 95, 375, ':', ':', (Validator) null, "Switch.already.set", linkedList0, (-195385983));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(argumentImpl0, linkedList0);
      writeableCommandLineImpl0.addSwitch(argumentImpl0, true);
      // Undeclared exception!
      try { 
        writeableCommandLineImpl0.addSwitch(argumentImpl0, true);
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
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      writeableCommandLineImpl0.addSwitch(propertyOption0, false);
      Boolean boolean0 = new Boolean("-D");
      Boolean boolean1 = writeableCommandLineImpl0.getSwitch((Option) propertyOption0, boolean0);
      assertFalse(boolean1);
      assertNotNull(boolean1);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<DefaultOption> linkedList0 = new LinkedList<DefaultOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = writeableCommandLineImpl0.hasOption("-D");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      writeableCommandLineImpl0.addValue(propertyOption0, propertyOption0);
      List list0 = writeableCommandLineImpl0.getValues((Option) propertyOption0, (List) null);
      assertNotNull(list0);
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      List list0 = writeableCommandLineImpl0.getValues((Option) propertyOption0, (List) null);
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      Properties properties0 = new Properties();
      writeableCommandLineImpl0.addValue(propertyOption0, properties0);
      List list0 = writeableCommandLineImpl0.getUndefaultedValues(propertyOption0);
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      List list0 = writeableCommandLineImpl0.getUndefaultedValues(propertyOption0);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      Boolean boolean0 = writeableCommandLineImpl0.getSwitch((Option) propertyOption0);
      assertNull(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption("]", "'aFCR!&/Kq=&h", 1523);
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addProperty((Option) propertyOption0, "-D", "]");
      writeableCommandLineImpl0.addProperty((Option) propertyOption0, "-D", "'aFCR!&/Kq=&h");
      assertEquals("'aFCR!&/Kq=&h", propertyOption0.getDescription());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      writeableCommandLineImpl0.addProperty((Option) propertyOption0, "-D", "Passes properties and values to the application");
      String string0 = writeableCommandLineImpl0.getProperty("Passes properties and values to the application");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      writeableCommandLineImpl0.addProperty((Option) propertyOption0, "-D", "Passes properties and values to the application");
      Set set0 = writeableCommandLineImpl0.getProperties();
      assertEquals(1, set0.size());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      boolean boolean0 = writeableCommandLineImpl0.looksLikeOption("Passes properties and values to the application");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      boolean boolean0 = writeableCommandLineImpl0.looksLikeOption("-D");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      linkedList0.push("?E (c<?R*G+}");
      String string0 = writeableCommandLineImpl0.toString();
      assertEquals("\"?E (c<?R*G+}\"", string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption("", "", 1523);
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      linkedList0.add((Object) "");
      linkedList0.push("");
      String string0 = writeableCommandLineImpl0.toString();
      assertEquals(" ", string0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      LinkedList<String> linkedList0 = new LinkedList<String>();
      writeableCommandLineImpl0.setDefaultValues(propertyOption0, linkedList0);
      assertEquals(68, propertyOption0.getId());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      writeableCommandLineImpl0.setDefaultValues(propertyOption0, (List) null);
      assertEquals("Passes properties and values to the application", propertyOption0.getDescription());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      Boolean boolean0 = Boolean.FALSE;
      writeableCommandLineImpl0.setDefaultSwitch(propertyOption0, boolean0);
      assertEquals(68, propertyOption0.getId());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      writeableCommandLineImpl0.setDefaultSwitch(propertyOption0, (Boolean) null);
      assertEquals("Passes properties and values to the application", propertyOption0.getDescription());
  }
}
