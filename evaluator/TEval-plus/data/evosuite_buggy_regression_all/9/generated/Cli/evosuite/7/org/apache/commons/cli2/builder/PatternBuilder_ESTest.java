/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:34:27 GMT 2023
 */

package org.apache.commons.cli2.builder;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.PatternBuilder;
import org.apache.commons.cli2.option.GroupImpl;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class PatternBuilder_ESTest extends PatternBuilder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PatternBuilder patternBuilder0 = new PatternBuilder();
      patternBuilder0.withPattern(":j*");
      Option option0 = patternBuilder0.create();
      assertFalse(option0.isRequired());
      assertEquals("-j", option0.getPreferredName());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PatternBuilder patternBuilder0 = new PatternBuilder();
      patternBuilder0.withPattern("[OzC");
      GroupImpl groupImpl0 = (GroupImpl)patternBuilder0.create();
      assertEquals(0, groupImpl0.getMinimum());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PatternBuilder patternBuilder0 = new PatternBuilder();
      // Undeclared exception!
      try { 
        patternBuilder0.withPattern("&<(g>_%!E=h\b:");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Cannot use null as a validator.
         //
         verifyException("org.apache.commons.cli2.builder.ArgumentBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PatternBuilder patternBuilder0 = new PatternBuilder();
      patternBuilder0.withPattern("EIBv7|?7&T\"cR w");
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PatternBuilder patternBuilder0 = new PatternBuilder();
      patternBuilder0.withPattern("+},rzjV?W#=j_$TWz(");
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PatternBuilder patternBuilder0 = new PatternBuilder();
      patternBuilder0.withPattern("hv'fFLS~0@");
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PatternBuilder patternBuilder0 = new PatternBuilder();
      patternBuilder0.withPattern("m`d)2HNtylGy[V|TdvV");
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PatternBuilder patternBuilder0 = new PatternBuilder();
      patternBuilder0.withPattern("+-$n4");
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PatternBuilder patternBuilder0 = new PatternBuilder();
      patternBuilder0.withPattern("ArgumentBuilder.null.consume.remaining");
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PatternBuilder patternBuilder0 = new PatternBuilder();
      patternBuilder0.withPattern("933dZT2A_m/F>m{");
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PatternBuilder patternBuilder0 = new PatternBuilder();
      patternBuilder0.withPattern("0#$5v]ju4s;9BrZJ1xg");
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PatternBuilder patternBuilder0 = new PatternBuilder();
      patternBuilder0.withPattern("G*VV<#Kd86A");
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PatternBuilder patternBuilder0 = new PatternBuilder();
      patternBuilder0.withPattern("");
  }
}
