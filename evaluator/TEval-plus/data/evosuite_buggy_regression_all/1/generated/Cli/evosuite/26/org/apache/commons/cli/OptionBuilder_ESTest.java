/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:00:10 GMT 2023
 */

package org.apache.commons.cli;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class OptionBuilder_ESTest extends OptionBuilder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      OptionBuilder optionBuilder0 = OptionBuilder.hasArgs(0);
      assertNotNull(optionBuilder0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      OptionBuilder optionBuilder0 = OptionBuilder.withValueSeparator('Z');
      OptionBuilder optionBuilder1 = OptionBuilder.withType(optionBuilder0);
      assertSame(optionBuilder0, optionBuilder1);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      OptionBuilder optionBuilder0 = OptionBuilder.isRequired();
      assertNotNull(optionBuilder0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      OptionBuilder optionBuilder0 = OptionBuilder.hasArg();
      assertNotNull(optionBuilder0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      OptionBuilder.withLongOpt("arg");
      Option option0 = OptionBuilder.create();
      assertEquals((-1), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      OptionBuilder optionBuilder0 = OptionBuilder.isRequired(true);
      assertNotNull(optionBuilder0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      OptionBuilder optionBuilder0 = OptionBuilder.withArgName("arg");
      assertNotNull(optionBuilder0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      OptionBuilder optionBuilder0 = OptionBuilder.withValueSeparator();
      assertNotNull(optionBuilder0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      OptionBuilder optionBuilder0 = OptionBuilder.hasOptionalArg();
      assertNotNull(optionBuilder0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Option option0 = OptionBuilder.create('i');
      assertEquals("i", option0.getOpt());
      assertEquals((-1), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      OptionBuilder optionBuilder0 = OptionBuilder.hasArgs();
      assertNotNull(optionBuilder0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      OptionBuilder optionBuilder0 = OptionBuilder.hasOptionalArgs((-1));
      assertNotNull(optionBuilder0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      OptionBuilder optionBuilder0 = OptionBuilder.hasOptionalArgs();
      assertNotNull(optionBuilder0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      OptionBuilder optionBuilder0 = OptionBuilder.withDescription("<bsJHYj;WSMYG&0+");
      assertNotNull(optionBuilder0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      OptionBuilder optionBuilder0 = OptionBuilder.hasArg(false);
      assertNotNull(optionBuilder0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      OptionBuilder optionBuilder0 = OptionBuilder.hasArg(true);
      assertNotNull(optionBuilder0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      try { 
        OptionBuilder.create();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must specify longopt
         //
         verifyException("org.apache.commons.cli.OptionBuilder", e);
      }
  }
}