/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:22:04 GMT 2023
 */

package org.apache.commons.cli2.option;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.time.ZoneId;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Set;
import org.apache.commons.cli2.Argument;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.WriteableCommandLine;
import org.apache.commons.cli2.commandline.WriteableCommandLineImpl;
import org.apache.commons.cli2.option.ArgumentImpl;
import org.apache.commons.cli2.option.Command;
import org.apache.commons.cli2.option.DefaultOption;
import org.apache.commons.cli2.option.GroupImpl;
import org.apache.commons.cli2.option.PropertyOption;
import org.apache.commons.cli2.option.SourceDestArgument;
import org.apache.commons.cli2.validation.DateValidator;
import org.apache.commons.cli2.validation.FileValidator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class OptionImpl_ESTest extends OptionImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      FileValidator fileValidator0 = FileValidator.getExistingDirectoryInstance();
      LinkedList<SourceDestArgument> linkedList0 = new LinkedList<SourceDestArgument>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("B|3", "Passes properties and values to the application", (-1262), (-1262), '^', 'K', fileValidator0, "Passes properties and values to the application", linkedList0, (-1262));
      Set set0 = argumentImpl0.getPrefixes();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "NTf65A|", "Passes properties and values to the applicationPasses properties and values to the application", (-1262), (-1262));
      Command command0 = new Command("B|3", "Passes properties and values to the applicationPasses properties and values to the application", set0, false, argumentImpl0, groupImpl0, (-467907902));
      assertFalse(command0.isRequired());
      
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(argumentImpl0, linkedList0);
      command0.validate(writeableCommandLineImpl0);
      assertEquals((-467907902), command0.getId());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      propertyOption0.defaults((WriteableCommandLine) null);
      assertEquals("Passes properties and values to the application", propertyOption0.getDescription());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      boolean boolean0 = propertyOption0.equals(propertyOption0);
      assertTrue(boolean0);
      assertEquals(68, propertyOption0.getId());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      propertyOption0.toString();
      assertEquals(68, propertyOption0.getId());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<SourceDestArgument> linkedList0 = new LinkedList<SourceDestArgument>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      LinkedList<Integer> linkedList1 = new LinkedList<Integer>();
      ListIterator<Integer> listIterator0 = linkedList1.listIterator();
      boolean boolean0 = propertyOption0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, (ListIterator) listIterator0);
      assertFalse(boolean0);
      assertEquals(68, propertyOption0.getId());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      ListIterator<String> listIterator0 = (ListIterator<String>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true).when(listIterator0).hasNext();
      doReturn("N_]rFWY`}<[%^2L+N_]rFWY`}<[%^2L+SF02~").when(listIterator0).next();
      doReturn("Passes properties and values to the application").when(listIterator0).previous();
      propertyOption0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, (ListIterator) listIterator0);
      assertEquals(68, propertyOption0.getId());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      boolean boolean0 = propertyOption0.equals("-D");
      assertEquals(68, propertyOption0.getId());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      PropertyOption propertyOption1 = new PropertyOption("Passes properties and values to the application", "Passes properties and values to the application", 631);
      boolean boolean0 = propertyOption1.equals(propertyOption0);
      assertEquals(631, propertyOption1.getId());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      DateValidator dateValidator0 = new DateValidator();
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl((String) null, "mif'x3Er.", 1, 1, 'Q', 'C', dateValidator0, (String) null, linkedList0, 68);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Q[dLMpx!\\>ekr,", "mif'x3Er.", 1, 1);
      Command command0 = new Command("Passes properties and values to the application", "-D!vF`wa0N\"", (Set) null, false, sourceDestArgument0, groupImpl0, 68);
      boolean boolean0 = propertyOption0.equals(command0);
      assertEquals(68, command0.getId());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      PropertyOption propertyOption1 = new PropertyOption("-D", "N6", 68);
      boolean boolean0 = propertyOption0.equals(propertyOption1);
      assertEquals(68, propertyOption1.getId());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption((String) null, "Passes properties and values to the application", (-1));
      propertyOption0.hashCode();
      assertEquals((-1), propertyOption0.getId());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      FileValidator fileValidator0 = FileValidator.getExistingFileInstance();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("kw?5p/[td7$&Bnq", "/!ic O.Z?JB+i?SHr", 0, 0, 'D', 'F', fileValidator0, "kw?5p/[td7$&Bnq", (List) null, (-1833));
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(sourceDestArgument0, (List) null);
      sourceDestArgument0.validate((WriteableCommandLine) writeableCommandLineImpl0);
      assertEquals((-1833), argumentImpl0.getId());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      Option option0 = propertyOption0.findOption("org.apache.commons.cli2.option.OptionImpl");
      assertNull(option0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      Option option0 = propertyOption0.findOption("-D");
      assertNotNull(option0);
      assertEquals(68, option0.getId());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      LinkedList<GroupImpl> linkedList0 = new LinkedList<GroupImpl>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "Passes properties and values to the application", "-D", (-458), (-458));
      DefaultOption defaultOption0 = new DefaultOption("Passes properties and values to the application", "-D", true, "Passes properties and values to the application", "Passes properties and values to the application", (Set) null, (Set) null, true, (Argument) null, groupImpl0, 1953);
      assertEquals(1953, defaultOption0.getId());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      FileValidator fileValidator0 = FileValidator.getExistingDirectoryInstance();
      LinkedList<SourceDestArgument> linkedList0 = new LinkedList<SourceDestArgument>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("B|3", "Passes properties and values to the application", (-1262), (-1262), '^', 'K', fileValidator0, "Passes properties and values to the application", linkedList0, (-1262));
      Set set0 = argumentImpl0.getPrefixes();
      argumentImpl0.checkPrefixes(set0);
      assertEquals((-1262), argumentImpl0.getId());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      Set<String> set0 = ZoneId.getAvailableZoneIds();
      // Undeclared exception!
      try { 
        propertyOption0.checkPrefixes(set0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Trigger -D must be prefixed with a value from java.util.HashSet@0000000003
         //
         verifyException("org.apache.commons.cli2.option.OptionImpl", e);
      }
  }
}
