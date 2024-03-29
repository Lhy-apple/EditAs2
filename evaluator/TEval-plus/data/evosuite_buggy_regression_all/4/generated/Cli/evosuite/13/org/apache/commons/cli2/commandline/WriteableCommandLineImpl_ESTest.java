/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:03:07 GMT 2023
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
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class WriteableCommandLineImpl_ESTest extends WriteableCommandLineImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      String string0 = writeableCommandLineImpl0.getProperty("-D");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      Set set0 = writeableCommandLineImpl0.getOptionTriggers();
      assertEquals(0, set0.size());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addProperty("-D", "Passes properties and values to the application");
      assertFalse(linkedList0.contains("-D"));
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      Set set0 = writeableCommandLineImpl0.getProperties();
      assertTrue(set0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      int int0 = writeableCommandLineImpl0.getOptionCount((Option) propertyOption0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      List list0 = writeableCommandLineImpl0.getNormalised();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addValue(propertyOption0, propertyOption0);
      writeableCommandLineImpl0.addValue(propertyOption0, propertyOption0);
      assertEquals("Passes properties and values to the application", propertyOption0.getDescription());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      DateValidator dateValidator0 = DateValidator.getTimeInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("-D", "GNt`b", (-122683469), (-122683469), 'd', 'd', dateValidator0, "Ar", linkedList0, (-122683469));
      writeableCommandLineImpl0.addValue(argumentImpl0, "GNt`b");
      assertEquals((-122683469), argumentImpl0.getId());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addSwitch(propertyOption0, false);
      boolean boolean0 = writeableCommandLineImpl0.hasOption((Option) propertyOption0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
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
  public void test10()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      writeableCommandLineImpl0.addSwitch(propertyOption0, true);
      assertEquals("Passes properties and values to the application", propertyOption0.getDescription());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      boolean boolean0 = writeableCommandLineImpl0.hasOption("-D");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      List list0 = writeableCommandLineImpl0.getValues((Option) propertyOption0, (List) null);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      Boolean boolean0 = Boolean.TRUE;
      writeableCommandLineImpl0.addSwitch(propertyOption0, false);
      Boolean boolean1 = writeableCommandLineImpl0.getSwitch((Option) propertyOption0, boolean0);
      assertFalse(boolean1);
      assertNotNull(boolean1);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      Boolean boolean0 = writeableCommandLineImpl0.getSwitch("j>6_%w");
      assertNull(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<DefaultOption> linkedList0 = new LinkedList<DefaultOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = writeableCommandLineImpl0.looksLikeOption("j{9");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      boolean boolean0 = writeableCommandLineImpl0.looksLikeOption("-D");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addValue(propertyOption0, "Passes properties and values to the application");
      List list0 = writeableCommandLineImpl0.getValues((Option) propertyOption0, (List) linkedList0);
      WriteableCommandLineImpl writeableCommandLineImpl1 = new WriteableCommandLineImpl(propertyOption0, list0);
      String string0 = writeableCommandLineImpl1.toString();
      assertEquals("\"Passes properties and values to the application\"", string0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      linkedList0.add((Object) "");
      String string0 = writeableCommandLineImpl0.toString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      linkedList0.add((Object) "");
      LinkedList<ArgumentImpl> linkedList1 = new LinkedList<ArgumentImpl>();
      linkedList0.add((Object) linkedList1);
      // Undeclared exception!
      try { 
        writeableCommandLineImpl0.toString();
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
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      List list0 = writeableCommandLineImpl0.getValues("", (List) linkedList0);
      writeableCommandLineImpl0.setDefaultValues((Option) null, list0);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      writeableCommandLineImpl0.setDefaultValues(propertyOption0, (List) null);
      assertEquals("Passes properties and values to the application", propertyOption0.getDescription());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      Boolean boolean0 = Boolean.FALSE;
      writeableCommandLineImpl0.setDefaultSwitch(propertyOption0, boolean0);
      assertEquals("-D", propertyOption0.getPreferredName());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.setDefaultSwitch(propertyOption0, (Boolean) null);
      assertEquals("Passes properties and values to the application", propertyOption0.getDescription());
  }
}
