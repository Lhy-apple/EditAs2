/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:21:47 GMT 2023
 */

package org.apache.commons.cli;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Collection;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionGroup;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class OptionGroup_ESTest extends OptionGroup_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      OptionGroup optionGroup0 = new OptionGroup();
      boolean boolean0 = optionGroup0.isRequired();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      OptionGroup optionGroup0 = new OptionGroup();
      optionGroup0.setRequired(true);
      assertTrue(optionGroup0.isRequired());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      OptionGroup optionGroup0 = new OptionGroup();
      Collection collection0 = optionGroup0.getNames();
      assertNotNull(collection0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("7OYCJ0w", "7OYCJ0w");
      optionGroup0.setSelected(option0);
      optionGroup0.setSelected(option0);
      assertEquals((-1), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      OptionGroup optionGroup0 = new OptionGroup();
      optionGroup0.setSelected((Option) null);
      assertFalse(optionGroup0.isRequired());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("7OYCJ0w", "7OYCJ0w");
      optionGroup0.setSelected(option0);
      Option option1 = new Option((String) null, "[-7OYCJ0w 7OYCJ0w]");
      try { 
        optionGroup0.setSelected(option1);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // The option 'null' was specified but an option from this group has already been selected: '7OYCJ0w'
         //
         verifyException("org.apache.commons.cli.OptionGroup", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("", false, "");
      optionGroup0.addOption(option0);
      Option option1 = new Option((String) null, true, "YP");
      optionGroup0.addOption(option1);
      String string0 = optionGroup0.toString();
      assertEquals("[- , --null YP]", string0);
  }
}