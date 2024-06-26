/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 12:26:31 GMT 2023
 */

package org.apache.commons.cli;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Collection;
import java.util.List;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionGroup;
import org.apache.commons.cli.Options;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Options_ESTest extends Options_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Options options0 = new Options();
      options0.addOption((String) null, "'");
      Option option0 = options0.getOption((String) null);
      assertNotNull(option0);
      assertFalse(option0.hasArg());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Options options0 = new Options();
      Collection<Option> collection0 = options0.getOptions();
      assertNotNull(collection0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Options options0 = new Options();
      List list0 = options0.getRequiredOptions();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Options options0 = new Options();
      String string0 = options0.toString();
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Options options0 = new Options();
      // Undeclared exception!
      try { 
        options0.getOptionGroup((Option) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.cli.Options", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Options options0 = new Options();
      Collection<OptionGroup> collection0 = options0.getOptionGroups();
      assertNotNull(collection0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      optionGroup0.setRequired(true);
      Options options1 = options0.addOptionGroup(optionGroup0);
      assertSame(options0, options1);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("N", (String) null);
      OptionGroup optionGroup1 = optionGroup0.addOption(option0);
      options0.addOptionGroup(optionGroup1);
      assertFalse(option0.isRequired());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Options options0 = new Options();
      options0.addOption("Z1c7itJTvuU", "'zk9feGR`Ls~4:", true, "org.apache.commons.cli.Options");
      boolean boolean0 = options0.hasOption("'zk9feGR`Ls~4:");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Options options0 = new Options();
      Option option0 = new Option((String) null, "aA>*?<E4\";\"_iGxN5");
      option0.setRequired(true);
      Options options1 = options0.addOption(option0);
      Options options2 = options1.addOption(option0);
      assertSame(options1, options2);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Options options0 = new Options();
      Option option0 = options0.getOption("ZgPO*,*x8P$Q]3X.<3");
      assertNull(option0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Options options0 = new Options();
      Options options1 = options0.addOption("I", "I", true, "I");
      List<String> list0 = options1.getMatchingOptions("I");
      assertTrue(list0.contains("I"));
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Options options0 = new Options();
      options0.addOption("", "aA>*?<E4\";\"_iGxN5", false, "aA>*?<E4\";\"_iGxN5");
      List<String> list0 = options0.getMatchingOptions("");
      assertTrue(list0.contains("aA>*?<E4\";\"_iGxN5"));
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Options options0 = new Options();
      Options options1 = options0.addOption("", "aA>*?<E4\";\"_iGxN5", false, "aA>*?<E4\";\"_iGxN5");
      List<String> list0 = options1.getMatchingOptions("3]Jw?");
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Options options0 = new Options();
      boolean boolean0 = options0.hasOption("");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Options options0 = new Options();
      Option option0 = new Option((String) null, "aA>*?<E4\";\"_iGxN5");
      Options options1 = options0.addOption(option0);
      boolean boolean0 = options1.hasOption((String) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Options options0 = new Options();
      boolean boolean0 = options0.hasLongOption("");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Options options0 = new Options();
      Options options1 = options0.addOption("", "", false, "");
      boolean boolean0 = options1.hasLongOption("");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Options options0 = new Options();
      boolean boolean0 = options0.hasShortOption("*>O?GC*@?gdgJz--(");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Options options0 = new Options();
      options0.addOption("", false, "5>T3");
      boolean boolean0 = options0.hasShortOption("");
      assertTrue(boolean0);
  }
}
