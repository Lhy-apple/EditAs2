/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:36:14 GMT 2023
 */

package org.apache.commons.codec.language;

import org.junit.Test;
import static org.junit.Assert.*;
import org.apache.commons.codec.language.RefinedSoundex;
import org.apache.commons.codec.language.SoundexUtils;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class SoundexUtils_ESTest extends SoundexUtils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      SoundexUtils soundexUtils0 = new SoundexUtils();
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      String string0 = SoundexUtils.clean((String) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      String string0 = SoundexUtils.clean("");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      RefinedSoundex refinedSoundex0 = RefinedSoundex.US_ENGLISH;
      int int0 = SoundexUtils.difference(refinedSoundex0, (String) null, "/CcYS@'(MW#~C_H1");
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      RefinedSoundex refinedSoundex0 = new RefinedSoundex();
      int int0 = SoundexUtils.difference(refinedSoundex0, "ARCHIT", (String) null);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      int int0 = SoundexUtils.differenceEncoded("h\"h[{sdBB", "ARCHIT");
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      int int0 = SoundexUtils.differenceEncoded("ZwhEnG", "ZwhEnG");
      assertEquals(6, int0);
  }
}
