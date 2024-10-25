/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:02:19 GMT 2023
 */

package org.apache.commons.cli;

import org.junit.Test;
import static org.junit.Assert.*;
import java.util.Iterator;
import java.util.List;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CommandLine_ESTest extends CommandLine_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      CommandLine commandLine0 = new CommandLine();
      boolean boolean0 = commandLine0.hasOption('I');
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CommandLine commandLine0 = new CommandLine();
      Option[] optionArray0 = commandLine0.getOptions();
      assertEquals(0, optionArray0.length);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CommandLine commandLine0 = new CommandLine();
      Iterator iterator0 = commandLine0.iterator();
      assertNotNull(iterator0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CommandLine commandLine0 = new CommandLine();
      String[] stringArray0 = commandLine0.getOptionValues('l');
      assertNull(stringArray0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CommandLine commandLine0 = new CommandLine();
      String string0 = commandLine0.getOptionValue('r');
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CommandLine commandLine0 = new CommandLine();
      String string0 = commandLine0.getOptionValue('=', (String) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CommandLine commandLine0 = new CommandLine();
      List list0 = commandLine0.getArgList();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CommandLine commandLine0 = new CommandLine();
      Object object0 = commandLine0.getOptionObject('L');
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CommandLine commandLine0 = new CommandLine();
      String[] stringArray0 = commandLine0.getArgs();
      assertEquals(0, stringArray0.length);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CommandLine commandLine0 = new CommandLine();
      commandLine0.addArg("                                                                ");
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CommandLine commandLine0 = new CommandLine();
      Option option0 = new Option("", true, "");
      commandLine0.addOption(option0);
      boolean boolean0 = commandLine0.hasOption("");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CommandLine commandLine0 = new CommandLine();
      Option option0 = new Option("", true, "!J2H~W%j9^");
      commandLine0.addOption(option0);
      option0.addValue("e#[|UVc^$)h");
      Object object0 = commandLine0.getOptionObject("");
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CommandLine commandLine0 = new CommandLine();
      Option option0 = new Option("", true, "!J2H~W%j9^");
      option0.setLongOpt("");
      commandLine0.addOption(option0);
      Object object0 = commandLine0.getOptionObject("");
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CommandLine commandLine0 = new CommandLine();
      Option option0 = new Option("", true, "");
      option0.addValue("org.apache.commons.cli.CommandLine");
      commandLine0.addOption(option0);
      String string0 = commandLine0.getOptionValue("", "O(m8=f");
      assertEquals("org.apache.commons.cli.CommandLine", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CommandLine commandLine0 = new CommandLine();
      Option option0 = new Option((String) null, "9E/yPmP(+[,9A");
      commandLine0.addOption(option0);
      assertEquals("arg", option0.getArgName());
  }
}
