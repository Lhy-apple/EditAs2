/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:35:57 GMT 2023
 */

package org.apache.commons.cli2.option;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.nio.charset.Charset;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.ListIterator;
import java.util.Locale;
import java.util.Set;
import org.apache.commons.cli2.Argument;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.WriteableCommandLine;
import org.apache.commons.cli2.commandline.WriteableCommandLineImpl;
import org.apache.commons.cli2.option.ArgumentImpl;
import org.apache.commons.cli2.option.Command;
import org.apache.commons.cli2.option.GroupImpl;
import org.apache.commons.cli2.option.PropertyOption;
import org.apache.commons.cli2.validation.NumberValidator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class OptionImpl_ESTest extends OptionImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      LinkedHashSet<Object> linkedHashSet0 = new LinkedHashSet<Object>();
      LinkedList<Locale> linkedList0 = new LinkedList<Locale>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "Passes properties and values to the application", (-2917), (-2917));
      Command command0 = new Command("-D", "Passes properties and values to the application", linkedHashSet0, true, (Argument) null, groupImpl0, (-1235450784));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      try { 
        command0.validate(writeableCommandLineImpl0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing required option -D
         //
         verifyException("org.apache.commons.cli2.option.Command", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      propertyOption0.defaults((WriteableCommandLine) null);
      assertFalse(propertyOption0.isRequired());
      assertEquals(68, propertyOption0.getId());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      boolean boolean0 = propertyOption0.equals(propertyOption0);
      assertFalse(propertyOption0.isRequired());
      assertEquals(68, propertyOption0.getId());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      propertyOption0.toString();
      assertFalse(propertyOption0.isRequired());
      assertEquals(68, propertyOption0.getId());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      LinkedList<GroupImpl> linkedList1 = new LinkedList<GroupImpl>();
      ListIterator<GroupImpl> listIterator0 = linkedList1.listIterator();
      boolean boolean0 = propertyOption0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, (ListIterator) listIterator0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      LinkedHashSet<ArgumentImpl> linkedHashSet0 = new LinkedHashSet<ArgumentImpl>();
      NumberValidator numberValidator0 = NumberValidator.getNumberInstance();
      LinkedList<Command> linkedList0 = new LinkedList<Command>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("org.apache.commons.li2.commandlin!.CommandLineImpl", "TJJ", 0, 0, '>', '>', numberValidator0, "@X|k\"({*~<7.~$&-", linkedList0, 0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "rD2`XP$7yc{7W:|", "rg.apache.commons.cli2.validtion.ClassValidator", 0, 0);
      Command command0 = new Command("rD2`XP$7yc{7W:|", "--", linkedHashSet0, false, argumentImpl0, groupImpl0, 287);
      linkedList0.add(command0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      ListIterator<Command> listIterator0 = linkedList0.listIterator();
      // Undeclared exception!
      try { 
        argumentImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, (ListIterator) listIterator0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // org.apache.commons.cli2.option.Command cannot be cast to java.lang.String
         //
         verifyException("org.apache.commons.cli2.option.OptionImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      boolean boolean0 = propertyOption0.equals("Passes properties and values to the application");
      assertFalse(propertyOption0.isRequired());
      assertEquals(68, propertyOption0.getId());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      PropertyOption propertyOption1 = new PropertyOption("-D", "Passes properties and values to the application", 635);
      boolean boolean0 = propertyOption0.equals(propertyOption1);
      assertFalse(propertyOption1.isRequired());
      assertFalse(propertyOption1.equals((Object)propertyOption0));
      assertFalse(boolean0);
      assertEquals(635, propertyOption1.getId());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption("", "Passes properties and values to the application", (-2455));
      PropertyOption propertyOption1 = new PropertyOption("Unexpected.token", "Passes properties and values to the application", (-2455));
      boolean boolean0 = propertyOption0.equals(propertyOption1);
      assertFalse(boolean0);
      assertEquals((-2455), propertyOption1.getId());
      assertFalse(propertyOption1.isRequired());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption((String) null, (String) null, (-16));
      propertyOption0.hashCode();
      assertEquals((-16), propertyOption0.getId());
      assertFalse(propertyOption0.isRequired());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.getProperty(".-0~^");
      assertEquals(68, propertyOption0.getId());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      Option option0 = propertyOption0.findOption((String) null);
      assertNull(option0);
      assertEquals(68, propertyOption0.getId());
      assertFalse(propertyOption0.isRequired());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      Option option0 = propertyOption0.findOption("-D");
      assertFalse(option0.isRequired());
      assertNotNull(option0);
      assertEquals(68, option0.getId());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedHashSet<GroupImpl> linkedHashSet0 = new LinkedHashSet<GroupImpl>();
      propertyOption0.checkPrefixes(linkedHashSet0);
      assertEquals(68, propertyOption0.getId());
      assertFalse(propertyOption0.isRequired());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      Set set0 = propertyOption0.getTriggers();
      propertyOption0.checkPrefixes(set0);
      assertEquals(68, propertyOption0.getId());
      assertFalse(propertyOption0.isRequired());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      Charset charset0 = Charset.defaultCharset();
      Set<String> set0 = charset0.aliases();
      // Undeclared exception!
      try { 
        propertyOption0.checkPrefixes(set0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Trigger -D must be prefixed with a value from java.util.Collections$UnmodifiableSet@0000000002
         //
         verifyException("org.apache.commons.cli2.option.OptionImpl", e);
      }
  }
}